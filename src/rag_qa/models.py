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
    explain: bool = Field(
        default=False,
        description="Capture a per-stage retrieval trace (dense/BM25/RRF/rerank) for visualization.",
    )


class SourceChunk(BaseModel):
    """A retrieved source chunk with metadata."""

    text: str
    filename: str
    doc_id: str
    chunk_index: int
    score: float


class CandidateTrace(BaseModel):
    """A single chunk's position in one retrieval stage."""

    doc_id: str
    chunk_index: int
    filename: str
    rank: int = Field(..., description="1-indexed position within this stage")
    score: float = Field(..., description="Stage-native score (cosine / BM25 / RRF / cross-encoder)")
    text_preview: str = Field("", description="Leading slice of the chunk text for display")


class FusedCandidateTrace(BaseModel):
    """A chunk after RRF fusion, retaining where each lane ranked it."""

    doc_id: str
    chunk_index: int
    filename: str
    dense_rank: int | None = Field(None, description="Rank in the dense lane, if it appeared")
    bm25_rank: int | None = Field(None, description="Rank in the BM25 lane, if it appeared")
    rrf_score: float = Field(..., description="Summed reciprocal-rank-fusion score")
    rank: int = Field(..., description="1-indexed position after fusion")
    text_preview: str = ""


class RerankedCandidateTrace(BaseModel):
    """A chunk after cross-encoder reranking, showing how far it moved."""

    doc_id: str
    chunk_index: int
    filename: str
    rerank_score: float
    rank: int = Field(..., description="1-indexed position after rerank")
    previous_rank: int | None = Field(None, description="1-indexed position before rerank (post-fusion)")
    text_preview: str = ""


class RetrievalTrace(BaseModel):
    """End-to-end trace of how a query traverses the retrieval pipeline.

    Each list captures one stage so a client can visualize the flow:
    query -> dense lane + sparse lane -> RRF fusion -> rerank -> answer.
    """

    question: str
    dense: list[CandidateTrace] = Field(default_factory=list)
    bm25: list[CandidateTrace] = Field(default_factory=list)
    fused: list[FusedCandidateTrace] = Field(default_factory=list)
    reranked: list[RerankedCandidateTrace] = Field(default_factory=list)
    reranker_enabled: bool = True
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage wall-clock timings: dense, bm25, fuse, rerank, generate.",
    )


class QueryResponse(BaseModel):
    """Response from the query endpoint."""

    answer: str
    sources: list[SourceChunk]
    question: str
    model: str
    trace: RetrievalTrace | None = Field(
        default=None,
        description="Per-stage retrieval trace; present only when the request set explain=true.",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    qdrant_connected: bool
