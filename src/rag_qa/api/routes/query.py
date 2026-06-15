"""Query route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from rag_qa.api.routes.documents import get_pipeline
from rag_qa.models import QueryRequest, QueryResponse, SourceChunk
from rag_qa.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Answer a question using the ingested documents."""
    try:
        result = await pipeline.query(
            question=request.question,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    sources = [
        SourceChunk(
            text=s["text"],
            filename=s["filename"],
            doc_id=s["doc_id"],
            chunk_index=s["chunk_index"],
            score=s["score"],
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        question=result["question"],
        model=result.get("model", "unknown"),
    )
