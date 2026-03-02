"""Entropic compressor — sentence-level entropy filtering with no external deps.

Scores each sentence by information density using word frequency analysis,
then drops low-entropy sentences (boilerplate, repetition) until the
target compression ratio is reached.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from scout_ai.context.models import CompressedContext


class EntropicCompressor:
    """Sentence-level entropy filtering compressor.

    Removes low-information sentences (boilerplate, repetition) based on
    word frequency analysis. Higher-entropy sentences carry more unique
    information and are retained.

    No external dependencies — uses simple word frequency as a proxy for TF-IDF.
    """

    def __init__(self, min_tokens: int = 500) -> None:
        self._min_tokens = min_tokens

    def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressedContext:
        """Compress by removing low-entropy sentences.

        If the text is shorter than ``min_tokens`` (estimated as chars/4),
        returns it unchanged — compression of short text is counterproductive.
        """
        if not text:
            return CompressedContext(
                text="",
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                method="entropic",
            )

        # Skip compression for short text
        estimated_tokens = len(text) // 4
        if estimated_tokens < self._min_tokens:
            return CompressedContext(
                text=text,
                original_length=len(text),
                compressed_length=len(text),
                compression_ratio=1.0,
                method="entropic",
                metadata={"skipped": True, "reason": "below_min_tokens"},
            )

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return CompressedContext(
                text=text,
                original_length=len(text),
                compressed_length=len(text),
                compression_ratio=1.0,
                method="entropic",
            )

        # Compute word frequencies across the entire document
        all_words = _tokenize(text)
        word_freq = Counter(all_words)
        total_words = len(all_words)

        # Score each sentence by entropy
        scored = []
        for sent in sentences:
            entropy = _sentence_entropy(sent, word_freq, total_words)
            scored.append((entropy, sent))

        # Sort by entropy descending (highest information first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top sentences until we hit target_ratio of original length
        target_length = int(len(text) * target_ratio)
        kept: list[str] = []
        current_length = 0

        for _entropy, sent in scored:
            if current_length >= target_length:
                break
            kept.append(sent)
            current_length += len(sent)

        # Re-order kept sentences by their original position
        original_order = {sent: i for i, sent in enumerate(sentences)}
        kept.sort(key=lambda s: original_order.get(s, 0))

        compressed = " ".join(kept)
        return CompressedContext(
            text=compressed,
            original_length=len(text),
            compressed_length=len(compressed),
            compression_ratio=len(compressed) / len(text) if text else 1.0,
            method="entropic",
            metadata={"sentences_kept": len(kept), "sentences_total": len(sentences)},
        )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic."""
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization — lowercase, alpha-only."""
    return re.findall(r"[a-z]+", text.lower())


def _sentence_entropy(
    sentence: str,
    word_freq: Counter[str],
    total_words: int,
) -> float:
    """Compute information entropy of a sentence given corpus word frequencies.

    Uses inverse document frequency as a proxy: rare words contribute
    more entropy than common ones.
    """
    words = _tokenize(sentence)
    if not words:
        return 0.0

    entropy = 0.0
    for word in words:
        freq = word_freq.get(word, 1)
        # IDF-like score: -log(freq/total)
        prob = freq / total_words if total_words > 0 else 1.0
        entropy += -math.log(prob + 1e-10)

    # Normalize by sentence length to avoid bias toward long sentences
    return entropy / len(words)
