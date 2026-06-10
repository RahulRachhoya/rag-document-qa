"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from rag_qa.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Return a test Settings instance with safe defaults."""
    return Settings(
        groq_api_key="test-key",
        qdrant_url="",
        reranker_enabled=False,
        chunk_size=200,
        chunk_overlap=20,
    )
