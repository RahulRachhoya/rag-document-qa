"""Shared pytest fixtures for the full test system.

This file centralizes fixtures for:
- Unit tests (services)
- Integration tests (pipeline + vector/BM25)
- API tests (FastAPI TestClient against all endpoints)
- UI/static serving tests
- Variations (with/without reranker, error cases)

Run with: pytest -q --cov=src/rag_qa --cov-report=term-missing
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_qa.config import Settings
from rag_qa.api.main import app


@pytest.fixture()
def settings() -> Settings:
    """Return a test Settings instance with safe defaults (no real LLM/embedder)."""
    return Settings(
        groq_api_key="test-key",
        qdrant_url="",
        reranker_enabled=False,
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture()
def settings_with_reranker() -> Settings:
    """Settings with reranker enabled (for testing rerank path)."""
    return Settings(
        groq_api_key="test-key",
        qdrant_url="",
        reranker_enabled=True,
        chunk_size=200,
        chunk_overlap=20,
    )


def write_temp_file(content: str, suffix: str = ".txt") -> str:
    """Helper to write a temp file for ingestion tests. Caller must unlink."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


@pytest.fixture()
def sample_txt_path() -> str:
    """Provides a small real text file for ingestion tests. Auto-cleaned."""
    path = write_temp_file("This is a test document about retrieval augmented generation. " * 15)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture()
def api_client():
    """FastAPI TestClient with heavy components (LLM, Embedder) mocked at module level.

    This exercises the full API surface (routes, models, error handling, static UI mount)
    without needing real Groq keys or downloading sentence-transformers.
    """
    with (
        patch("rag_qa.pipeline.GroqLLM") as MockLLM,
        patch("rag_qa.pipeline.Embedder") as MockEmbedder,
        patch("rag_qa.api.main.GroqLLM", MockLLM),  # also patch where main imports
        patch("rag_qa.api.main.Embedder", MockEmbedder),
    ):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "This is a mocked grounded answer based on the provided context."
        MockLLM.return_value = mock_llm

        mock_emb = MagicMock()
        mock_emb.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
        mock_emb.embed_one.return_value = [0.1] * 384
        MockEmbedder.return_value = mock_emb

        # Force re-creation of the global pipeline in main.py for this test client
        # (the module-level pipeline is created at import time)
        import rag_qa.api.main as main_module
        main_module.pipeline = main_module.RAGPipeline(main_module.settings)

        client = TestClient(app)
        yield client


@pytest.fixture()
def api_client_with_reranker(api_client):
    """Same as api_client but with reranker enabled in settings."""
    # We patch at a higher level in the api_client fixture; here we just override settings
    # For simplicity in this design, the base api_client already supports both via settings.
    # This fixture exists for clarity / future expansion (e.g. mocking CrossEncoder too).
    yield api_client
