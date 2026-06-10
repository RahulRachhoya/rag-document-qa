"""Pydantic models for the RAG Q&A API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""

    doc_id: str
    filename: str
    chunk_count: int
    created_at: str


class IngestResponse(BaseModel):
    """Response from document ingestion endpoint."""

    doc_id: str
    filename: str
    chunks_created: int
    vectors_stored: int
    message: str = "Document ingested successfully"


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_ids: list[str] | None = Field(default=None, description="Filter to specific doc IDs")


class SourceChunk(BaseModel):
    """A retrieved source chunk with metadata."""

    text: str
    filename: str
    doc_id: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    """Response from the query endpoint."""

    answer: str
    sources: list[SourceChunk]
    question: str
    model: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    qdrant_connected: bool
