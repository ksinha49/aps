"""PageIndex tree index builder — the core ingestion provider.

Internalizes ``tree_parser()`` → ``meta_processor()`` → ``process_*()``
from vanilla PageIndex's ``page_index.py``, with:
- Pre-OCR'd input (no PDF parsing)
- Configurable LLM backend
- Medical-domain heuristic pre-pass
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pageindex_rag.aps.prompts import (
    CHECK_TITLE_APPEARANCE_PROMPT,
    CHECK_TITLE_START_PROMPT,
    GENERATE_SUMMARY_PROMPT,
    GENERATE_TOC_CONTINUE_PROMPT,
    GENERATE_TOC_INIT_PROMPT,
    TOC_DETECT_PROMPT,
)
from pageindex_rag.config import PageIndexSettings
from pageindex_rag.exceptions import IndexBuildError
from pageindex_rag.interfaces.ingestion import IIngestionProvider
from pageindex_rag.models import DocumentIndex, MedicalSectionType, PageContent, TreeNode
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.medical_classifier import MedicalSectionClassifier
from pageindex_rag.providers.pageindex.tokenizer import TokenCounter
from pageindex_rag.providers.pageindex.tree_builder import TreeBuilder
from pageindex_rag.providers.pageindex.tree_utils import (
    add_node_text,
    add_preface_if_needed,
    convert_physical_index_to_int,
    flatten_nodes,
    get_text_of_pages,
    validate_physical_indices,
    write_node_ids,
)

log = logging.getLogger(__name__)


class PageIndexIndexer(IIngestionProvider):
    """Build a hierarchical tree index from pre-OCR'd pages.

    Three indexing modes (same as vanilla PageIndex):
      1. TOC detected with page numbers → parse + verify
      2. TOC detected without page numbers → LLM maps sections to pages
      3. No TOC → LLM generates structure from content chunks

    When ``enable_medical_classification`` is True, a regex heuristic
    pre-pass detects sections before falling back to LLM generation.
    """

    def __init__(self, settings: PageIndexSettings, client: LLMClient) -> None:
        self._settings = settings
        self._client = client
        self._tc = TokenCounter(method=settings.tokenizer_method, model=settings.tokenizer_model)
        self._tree_builder = TreeBuilder(self._tc)
        self._classifier = MedicalSectionClassifier(client) if settings.enable_medical_classification else None

    async def build_index(
        self,
        pages: list[PageContent],
        doc_id: str,
        doc_name: str,
    ) -> DocumentIndex:
        """Build a ``DocumentIndex`` from pre-OCR'd pages."""
        if not pages:
            raise IndexBuildError("No pages provided")

        # Populate token counts if missing
        for p in pages:
            if p.token_count is None:
                p.token_count = self._tc.count(p.text)

        log.info(f"Building index for '{doc_name}' ({len(pages)} pages)")

        # Build the tree
        tree = await self._tree_parser(pages)

        # Assign node IDs
        write_node_ids(tree)

        # Populate node text
        add_node_text(tree, pages)

        # Generate summaries if enabled
        if self._settings.enable_node_summaries:
            await self._generate_summaries(tree)

        # Classify sections if enabled
        if self._classifier:
            await self._classify_nodes(tree)

        # Generate doc description if enabled
        doc_description = ""
        if self._settings.enable_doc_description:
            doc_description = await self._generate_doc_description(tree)

        return DocumentIndex(
            doc_id=doc_id,
            doc_name=doc_name,
            doc_description=doc_description,
            total_pages=len(pages),
            tree=tree,
            created_at=datetime.now(timezone.utc),
        )

    # ── Core tree building pipeline ──────────────────────────────────

    async def _tree_parser(self, pages: list[PageContent]) -> list[TreeNode]:
        """Main entry: detect TOC → meta_processor → post-process → split large nodes."""

        # Try medical heuristic first
        if self._classifier:
            heuristic_sections = self._classifier.detect_sections_heuristic(pages)
            if len(heuristic_sections) >= 3:
                log.info(f"Using heuristic sections ({len(heuristic_sections)} found)")
                toc_items = self._heuristic_to_toc(heuristic_sections, len(pages))
                toc_items = await self._check_title_appearances(toc_items, pages)
                valid = [i for i in toc_items if i.get("physical_index") is not None]
                if valid:
                    tree = self._tree_builder.build_tree(valid, len(pages))
                    tree = await self._process_large_nodes(tree, pages)
                    return tree

        # Fallback to LLM-based TOC detection
        toc_result = await self._check_toc(pages)

        if toc_result["toc_content"] and toc_result["page_index_given_in_toc"] == "yes":
            toc_items = await self._meta_processor(
                pages, mode="process_toc_with_page_numbers",
                toc_content=toc_result["toc_content"],
                toc_page_list=toc_result["toc_page_list"],
            )
        else:
            toc_items = await self._meta_processor(pages, mode="process_no_toc")

        toc_items = add_preface_if_needed(toc_items)
        toc_items = await self._check_title_appearances(toc_items, pages)
        valid = [i for i in toc_items if i.get("physical_index") is not None]

        tree = self._tree_builder.build_tree(valid, len(pages))
        tree = await self._process_large_nodes(tree, pages)
        return tree

    async def _meta_processor(
        self,
        pages: list[PageContent],
        mode: str,
        toc_content: Optional[str] = None,
        toc_page_list: Optional[list[int]] = None,
        start_index: int = 1,
    ) -> list[dict[str, Any]]:
        """Three-mode TOC processing with accuracy verification and fallback.

        Mirrors ``meta_processor()`` from vanilla PageIndex.
        """
        if mode == "process_toc_with_page_numbers":
            toc_items = await self._process_toc_with_page_numbers(
                toc_content or "", toc_page_list or [], pages, start_index
            )
        elif mode == "process_toc_no_page_numbers":
            toc_items = await self._process_toc_no_page_numbers(
                toc_content or "", pages, start_index
            )
        else:
            toc_items = await self._process_no_toc(pages, start_index)

        # Filter out None physical indices
        toc_items = [i for i in toc_items if i.get("physical_index") is not None]

        # Validate indices
        toc_items = validate_physical_indices(toc_items, len(pages), start_index)
        toc_items = [i for i in toc_items if i.get("physical_index") is not None]

        # Verify accuracy
        accuracy, incorrect = await self._verify_toc(pages, toc_items, start_index)
        log.info(f"TOC accuracy: {accuracy:.0%}, incorrect: {len(incorrect)}")

        if accuracy == 1.0:
            return toc_items

        if accuracy > 0.6 and incorrect:
            toc_items, _ = await self._fix_incorrect_toc(
                toc_items, pages, incorrect, start_index
            )
            return toc_items

        # Fallback cascade
        if mode == "process_toc_with_page_numbers":
            return await self._meta_processor(
                pages, "process_toc_no_page_numbers",
                toc_content=toc_content, toc_page_list=toc_page_list,
                start_index=start_index,
            )
        elif mode == "process_toc_no_page_numbers":
            return await self._meta_processor(pages, "process_no_toc", start_index=start_index)

        raise IndexBuildError("All indexing modes failed")

    # ── Mode implementations ─────────────────────────────────────────

    async def _process_no_toc(
        self, pages: list[PageContent], start_index: int = 1
    ) -> list[dict[str, Any]]:
        """Mode 3: No TOC found — LLM generates structure from chunks."""
        page_contents, token_lengths = self._prepare_labeled_pages(pages, start_index)
        group_texts = self._tree_builder.group_pages(
            page_contents, token_lengths, self._settings.max_group_tokens
        )

        toc = self._parse_json_response(
            await self._client.complete(
                GENERATE_TOC_INIT_PROMPT.format(part=group_texts[0])
            )
        )

        for group_text in group_texts[1:]:
            additional = self._parse_json_response(
                await self._client.complete(
                    GENERATE_TOC_CONTINUE_PROMPT.format(
                        previous_toc=json.dumps(toc, indent=2), part=group_text
                    )
                )
            )
            if isinstance(additional, list):
                toc.extend(additional)

        toc = convert_physical_index_to_int(toc)
        return toc

    async def _process_toc_with_page_numbers(
        self,
        toc_content: str,
        toc_page_list: list[int],
        pages: list[PageContent],
        start_index: int,
    ) -> list[dict[str, Any]]:
        """Mode 1: TOC with page numbers → transform + offset calculation."""
        toc_json = await self._toc_transformer(toc_content)

        # Find physical page offset using a sample of pages after TOC
        start_page = toc_page_list[-1] + 1 if toc_page_list else 0
        sample_content = get_text_of_pages(
            pages, start_page + 1,
            min(start_page + self._settings.toc_check_page_count, len(pages)),
            with_labels=True,
        )

        if sample_content:
            physical_indices = await self._extract_toc_indices(toc_json, sample_content)
            offset = self._calculate_page_offset(toc_json, physical_indices, start_page + 1)
            if offset is not None:
                for item in toc_json:
                    if item.get("page") is not None and isinstance(item["page"], int):
                        item["physical_index"] = item["page"] + offset
                        del item["page"]

        toc_json = convert_physical_index_to_int(toc_json)
        return toc_json

    async def _process_toc_no_page_numbers(
        self,
        toc_content: str,
        pages: list[PageContent],
        start_index: int,
    ) -> list[dict[str, Any]]:
        """Mode 2: TOC detected but no page numbers → LLM maps sections."""
        toc_json = await self._toc_transformer(toc_content)

        page_contents, token_lengths = self._prepare_labeled_pages(pages, start_index)
        group_texts = self._tree_builder.group_pages(
            page_contents, token_lengths, self._settings.max_group_tokens
        )

        toc_with_pages = copy.deepcopy(toc_json)
        for group_text in group_texts:
            toc_with_pages = await self._add_page_numbers_to_toc(group_text, toc_with_pages)

        toc_with_pages = convert_physical_index_to_int(toc_with_pages)
        return toc_with_pages

    # ── TOC detection ────────────────────────────────────────────────

    async def _check_toc(self, pages: list[PageContent]) -> dict[str, Any]:
        """Scan first N pages for a table of contents."""
        toc_page_list: list[int] = []
        last_was_toc = False
        check_limit = min(self._settings.toc_check_page_count, len(pages))

        for i in range(check_limit):
            if i >= check_limit and not last_was_toc:
                break

            prompt = TOC_DETECT_PROMPT.format(content=pages[i].text)
            response = await self._client.complete(prompt)
            parsed = self._client.extract_json(response)
            is_toc = parsed.get("toc_detected", "no") == "yes"

            if is_toc:
                toc_page_list.append(i)
                last_was_toc = True
            elif last_was_toc:
                break
            else:
                last_was_toc = False

        if not toc_page_list:
            return {"toc_content": None, "toc_page_list": [], "page_index_given_in_toc": "no"}

        toc_content = "".join(pages[i].text for i in toc_page_list)
        # Check if page numbers are present in TOC
        has_page_nums = await self._detect_page_numbers_in_toc(toc_content)

        return {
            "toc_content": toc_content,
            "toc_page_list": toc_page_list,
            "page_index_given_in_toc": has_page_nums,
        }

    async def _detect_page_numbers_in_toc(self, toc_content: str) -> str:
        prompt = f"""Detect if there are page numbers/indices in this table of contents.

Given text: {toc_content}

Return JSON:
{{"thinking": "<reasoning>", "page_index_given_in_toc": "<yes or no>"}}
Directly return JSON only."""
        response = await self._client.complete(prompt)
        parsed = self._client.extract_json(response)
        return parsed.get("page_index_given_in_toc", "no")

    async def _toc_transformer(self, toc_content: str) -> list[dict[str, Any]]:
        """Transform raw TOC text into structured JSON."""
        prompt = f"""Transform this table of contents into JSON format.

structure is the hierarchy index (1, 1.1, 1.2, 2, etc.).

Return JSON:
{{"table_of_contents": [
    {{"structure": "<x.x.x>", "title": "<section title>", "page": <page_number or null>}},
    ...
]}}

Given table of contents:
{toc_content}

Directly return the final JSON structure. Do not output anything else."""

        content, finish_reason = await self._client.complete_with_finish_reason(prompt)

        if finish_reason == "finished":
            parsed = self._client.extract_json(content)
            if isinstance(parsed, dict) and "table_of_contents" in parsed:
                return parsed["table_of_contents"]
            return parsed if isinstance(parsed, list) else []

        # Handle truncated output with continuation
        for _ in range(3):
            cont_prompt = f"""Continue the table of contents JSON structure.

Raw TOC: {toc_content}

Incomplete JSON so far: {content}

Continue directly from where it was cut off."""
            new_content, finish_reason = await self._client.complete_with_finish_reason(cont_prompt)
            content += new_content
            if finish_reason == "finished":
                break

        parsed = self._client.extract_json(content)
        if isinstance(parsed, dict) and "table_of_contents" in parsed:
            return parsed["table_of_contents"]
        return parsed if isinstance(parsed, list) else []

    async def _extract_toc_indices(
        self, toc_json: list[dict[str, Any]], content: str
    ) -> list[dict[str, Any]]:
        """Map TOC entries to physical page indices using document content."""
        toc_no_page = [{k: v for k, v in item.items() if k != "page"} for item in toc_json]
        prompt = f"""Add physical_index to each TOC entry based on where sections appear in the document pages.

Pages use <physical_index_X> tags. Only add physical_index for sections found in provided pages.

Table of contents: {json.dumps(toc_no_page, indent=2)}
Document pages: {content}

Return JSON array:
[{{"structure": "<>", "title": "<>", "physical_index": "<physical_index_X>"}}]

Directly return JSON only."""
        response = await self._client.complete(prompt)
        result = self._client.extract_json(response)
        return result if isinstance(result, list) else []

    async def _add_page_numbers_to_toc(
        self, part: str, structure: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fill in physical_index for TOC entries that appear in a page group."""
        prompt = f"""Check if sections from the structure appear in the document pages.

Pages use <physical_index_X> tags. Add physical_index for sections found.

Current Document Pages:
{part}

Given Structure:
{json.dumps(structure, indent=2)}

Return the full structure with physical_index added where found.
Directly return JSON only."""
        response = await self._client.complete(prompt)
        result = self._client.extract_json(response)
        return result if isinstance(result, list) else structure

    # ── Verification and fixing ──────────────────────────────────────

    async def _check_title_appearances(
        self, toc_items: list[dict[str, Any]], pages: list[PageContent]
    ) -> list[dict[str, Any]]:
        """Check if each TOC title actually appears on its assigned page."""
        tasks = []
        for item in toc_items:
            pi = item.get("physical_index")
            if pi is None or pi < 1 or pi > len(pages):
                item["appear_start"] = "no"
                continue
            page = pages[pi - 1]
            tasks.append(self._check_single_title_start(item["title"], page.text))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        task_idx = 0
        for item in toc_items:
            if item.get("appear_start") == "no":
                continue
            if task_idx < len(results):
                result = results[task_idx]
                item["appear_start"] = result if isinstance(result, str) else "no"
                task_idx += 1

        return toc_items

    async def _check_single_title_start(self, title: str, page_text: str) -> str:
        prompt = CHECK_TITLE_START_PROMPT.format(title=title, page_text=page_text)
        response = await self._client.complete(prompt)
        parsed = self._client.extract_json(response)
        return parsed.get("start_begin", "no")

    async def _verify_toc(
        self,
        pages: list[PageContent],
        toc_items: list[dict[str, Any]],
        start_index: int = 1,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Verify TOC accuracy by checking title appearances on assigned pages."""
        if not toc_items:
            return 0.0, []

        tasks = []
        indexed_items = []
        for idx, item in enumerate(toc_items):
            pi = item.get("physical_index")
            if pi is None:
                continue
            page_idx = pi - start_index
            if page_idx < 0 or page_idx >= len(pages):
                continue
            item_copy = {**item, "list_index": idx}
            tasks.append(self._check_title_on_page(item_copy, pages, start_index))
            indexed_items.append(item_copy)

        if not tasks:
            return 0.0, []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        correct = 0
        incorrect: list[dict[str, Any]] = []
        valid_count = 0

        for result in results:
            if isinstance(result, Exception):
                continue
            valid_count += 1
            if result.get("answer") == "yes":
                correct += 1
            else:
                incorrect.append(result)

        accuracy = correct / valid_count if valid_count > 0 else 0.0
        return accuracy, incorrect

    async def _check_title_on_page(
        self,
        item: dict[str, Any],
        pages: list[PageContent],
        start_index: int,
    ) -> dict[str, Any]:
        """Check if a title appears on its assigned physical page."""
        pi = item["physical_index"]
        page_idx = pi - start_index
        if page_idx < 0 or page_idx >= len(pages):
            return {"list_index": item.get("list_index"), "answer": "no", "title": item["title"]}

        page_text = pages[page_idx].text
        prompt = CHECK_TITLE_APPEARANCE_PROMPT.format(title=item["title"], page_text=page_text)
        response = await self._client.complete(prompt)
        parsed = self._client.extract_json(response)

        return {
            "list_index": item.get("list_index"),
            "answer": parsed.get("answer", "no"),
            "title": item["title"],
            "page_number": pi,
        }

    async def _fix_incorrect_toc(
        self,
        toc_items: list[dict[str, Any]],
        pages: list[PageContent],
        incorrect: list[dict[str, Any]],
        start_index: int,
        max_attempts: int = 3,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Attempt to fix incorrect TOC page assignments."""
        current_incorrect = incorrect
        for attempt in range(max_attempts):
            if not current_incorrect:
                break
            log.info(f"Fix attempt {attempt + 1}: {len(current_incorrect)} items")

            for item in current_incorrect:
                idx = item.get("list_index")
                if idx is None or idx < 0 or idx >= len(toc_items):
                    continue

                # Find search range (prev correct to next correct)
                prev_pi = start_index
                for j in range(idx - 1, -1, -1):
                    if toc_items[j].get("physical_index") is not None:
                        prev_pi = toc_items[j]["physical_index"]
                        break

                next_pi = len(pages) + start_index - 1
                for j in range(idx + 1, len(toc_items)):
                    if toc_items[j].get("physical_index") is not None:
                        next_pi = toc_items[j]["physical_index"]
                        break

                # Search in range
                search_content = get_text_of_pages(pages, prev_pi, next_pi, with_labels=True)
                prompt = f"""Find the physical page where this section starts.

Section Title: {item['title']}
Document pages: {search_content}

Return JSON: {{"physical_index": "<physical_index_X>"}}
Directly return JSON only."""
                response = await self._client.complete(prompt)
                parsed = self._client.extract_json(response)
                new_pi = convert_physical_index_to_int(parsed.get("physical_index"))
                if isinstance(new_pi, int):
                    toc_items[idx]["physical_index"] = new_pi

            # Re-verify
            _, current_incorrect = await self._verify_toc(pages, toc_items, start_index)

        return toc_items, current_incorrect

    # ── Large node splitting ─────────────────────────────────────────

    async def _process_large_nodes(
        self,
        tree: list[TreeNode],
        pages: list[PageContent],
        _depth: int = 0,
    ) -> list[TreeNode]:
        """Recursively split nodes that are too large."""
        if _depth >= self._settings.max_recursion_depth:
            return tree

        for node in tree:
            page_range = node.end_index - node.start_index + 1
            node_pages = [p for p in pages if node.start_index <= p.page_number <= node.end_index]
            token_count = sum(p.token_count or 0 for p in node_pages)

            if (
                page_range > self._settings.max_pages_per_node
                and token_count >= self._settings.max_tokens_per_node
                and not node.children
            ):
                log.info(
                    f"Splitting large node: {node.title} "
                    f"(pages {node.start_index}-{node.end_index}, {token_count} tokens)"
                )
                sub_toc = await self._process_no_toc(node_pages, node.start_index)
                sub_toc = await self._check_title_appearances(sub_toc, pages)
                valid = [i for i in sub_toc if i.get("physical_index") is not None]
                if valid:
                    node.children = self._tree_builder.build_tree(valid, node.end_index)

            if node.children:
                await self._process_large_nodes(node.children, pages, _depth + 1)

        return tree

    # ── Summary generation ───────────────────────────────────────────

    async def _generate_summaries(self, tree: list[TreeNode]) -> None:
        """Generate summaries for all nodes concurrently."""
        all_nodes = flatten_nodes(tree)
        sem = asyncio.Semaphore(self._settings.retrieval_max_concurrent)

        async def _summarize(node: TreeNode) -> None:
            if not node.text:
                return
            if self._tc.count(node.text) < 200:
                node.summary = node.text[:500]
                return
            async with sem:
                prompt = GENERATE_SUMMARY_PROMPT.format(text=node.text[:4000])
                node.summary = await self._client.complete(prompt)

        await asyncio.gather(*[_summarize(n) for n in all_nodes], return_exceptions=True)

    async def _classify_nodes(self, tree: list[TreeNode]) -> None:
        """Classify all nodes by medical section type."""
        if not self._classifier:
            return
        for node in flatten_nodes(tree):
            node.content_type = await self._classifier.classify(node.title, node.text[:500])

    async def _generate_doc_description(self, tree: list[TreeNode]) -> str:
        """Generate a one-sentence document description."""
        structure_summary = []
        for node in tree:
            structure_summary.append(
                f"- {node.title} (pp. {node.start_index}-{node.end_index})"
                + (f": {node.summary[:100]}" if node.summary else "")
            )
        prompt = f"""Generate a one-sentence description for this medical document (APS).

Document structure:
{chr(10).join(structure_summary)}

Return the description only."""
        return await self._client.complete(prompt)

    # ── Utility methods ──────────────────────────────────────────────

    def _prepare_labeled_pages(
        self, pages: list[PageContent], start_index: int = 1
    ) -> tuple[list[str], list[int]]:
        """Prepare page texts with physical_index labels and token counts."""
        contents: list[str] = []
        lengths: list[int] = []
        for p in pages:
            labeled = f"<physical_index_{p.page_number}>\n{p.text}\n<physical_index_{p.page_number}>\n\n"
            contents.append(labeled)
            lengths.append(self._tc.count(labeled))
        return contents, lengths

    def _parse_json_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response as JSON array."""
        result = self._client.extract_json(response)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "table_of_contents" in result:
            return result["table_of_contents"]
        return []

    def _heuristic_to_toc(
        self, sections: list[dict[str, Any]], total_pages: int
    ) -> list[dict[str, Any]]:
        """Convert heuristic section detections to TOC format."""
        toc: list[dict[str, Any]] = []
        for i, section in enumerate(sections):
            toc.append({
                "structure": str(i + 1),
                "title": section["title"],
                "physical_index": section["page_number"],
            })
        return toc

    @staticmethod
    def _calculate_page_offset(
        toc_with_pages: list[dict[str, Any]],
        physical_indices: list[dict[str, Any]],
        start_page_index: int,
    ) -> Optional[int]:
        """Calculate offset between logical page numbers and physical pages."""
        pairs: list[tuple[int, int]] = []
        pi_by_title = {item.get("title"): item.get("physical_index") for item in physical_indices}

        for item in toc_with_pages:
            title = item.get("title")
            page = item.get("page")
            pi = pi_by_title.get(title)
            if page is not None and pi is not None and isinstance(page, int) and isinstance(pi, int):
                if pi >= start_page_index:
                    pairs.append((pi, page))

        if not pairs:
            return None

        diffs = [pi - page for pi, page in pairs]
        # Most common difference
        from collections import Counter
        counts = Counter(diffs)
        return counts.most_common(1)[0][0]
