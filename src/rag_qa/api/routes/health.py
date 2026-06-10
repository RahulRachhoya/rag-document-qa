"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

from rag_qa import __version__
from rag_qa.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="ok",
        version=__version__,
        qdrant_connected=True,
    )
