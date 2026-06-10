"""Document management routes: upload, list, delete."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi import File as FastAPIFile

from rag_qa.models import DocumentInfo, IngestResponse
from rag_qa.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Shared pipeline instance (set by app startup)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised")
    return _pipeline


def set_pipeline(pipeline: RAGPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/upload", response_model=IngestResponse, status_code=201)
async def upload_document(
    file: UploadFile = FastAPIFile(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Upload and ingest a document (PDF, DOCX, or TXT)."""
    suffix = Path(file.filename or "file.txt").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    temp_path = upload_dir / f"{uuid.uuid4()}{suffix}"

    try:
        with temp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        size = temp_path.stat().st_size
        if size > MAX_SIZE_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large ({size} bytes). Max 20 MB.")

        result = await pipeline.ingest(str(temp_path), file.filename or "unknown")
        return IngestResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunks_created=result["chunks"],
            vectors_stored=result["vectors_stored"],
        )
    finally:
        if temp_path.exists():
            os.remove(temp_path)


@router.get("/", response_model=list[DocumentInfo])
async def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)) -> list[DocumentInfo]:
    """List all ingested documents."""
    docs = pipeline.list_documents()
    return [DocumentInfo(**d) for d in docs]


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> None:
    """Delete a document and its vectors."""
    deleted = pipeline.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
