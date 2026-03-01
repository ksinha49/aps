"""Async index creation endpoint — enqueues to SQS and returns immediately."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(tags=["indexing"])


class AsyncIndexRequest(BaseModel):
    """Request to enqueue an async indexing job."""

    doc_id: str
    doc_name: str
    pages: list[dict[str, object]] = Field(
        ..., description="List of {page_number, text} objects"
    )


class AsyncIndexResponse(BaseModel):
    """Response confirming the job was enqueued."""

    job_id: str
    status: str = "queued"
    message: str = "Indexing job has been queued for processing."


def _send_to_queue(queue_url: str, message: dict[str, object], region: str) -> str:
    """Send a message to an SQS queue and return the generated job ID.

    Parameters
    ----------
    queue_url:
        The SQS queue URL.
    message:
        The message payload to send (will be JSON-serialised).
    region:
        AWS region for the SQS client.

    Returns
    -------
    str
        A UUID job ID that was included in the message.
    """
    import boto3

    job_id = str(uuid.uuid4())
    message["job_id"] = job_id

    client = boto3.client("sqs", region_name=region)
    client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message, default=str),
    )
    log.info("Enqueued indexing job %s to %s", job_id, queue_url)
    return job_id


@router.post("/index/async", response_model=AsyncIndexResponse, status_code=202)
async def create_index_async(request: AsyncIndexRequest, req: Request) -> JSONResponse:
    """Enqueue an indexing job to SQS and return immediately with a job ID."""
    settings = req.app.state.settings
    queue_url: str = settings.index_queue_url

    if not queue_url:
        return JSONResponse(
            status_code=501,
            content={"detail": "Async indexing not configured. Set SCOUT_INDEX_QUEUE_URL."},
        )

    message: dict[str, object] = {
        "doc_id": request.doc_id,
        "doc_name": request.doc_name,
        "pages": request.pages,
    }

    job_id = _send_to_queue(
        queue_url=queue_url,
        message=message,
        region=settings.aws_region,
    )

    return JSONResponse(
        status_code=202,
        content=AsyncIndexResponse(job_id=job_id).model_dump(),
    )
