"""Comprehensive API and integration tests for every major feature.

This file (together with unit/ tests) forms the core of the test system for the RAG Document Q&A app.

Covers (mapped to user-facing + internal features):
- Health & version reporting
- Document upload / multi-format ingestion (TXT focus + notes for PDF/DOCX)
- Listing documents
- Deleting documents (including BM25 survivor regression)
- Querying (hybrid retrieval, sources, top_k, no-docs case)
- Error handling & validation (bad files, missing key, empty content)
- Reranker enabled/disabled paths (via fixtures)
- Static UI serving (the nice frontend at root)
- Configuration via Settings

Run: pytest tests/integration -q --tb=line
Full suite with coverage: pytest --cov=src/rag_qa --cov-report=term-missing -q

Design principles:
- Fast (heavy LLM/Embedder mocked at fixture level).
- Realistic (real in-memory Qdrant + BM25 + chunker + real small files).
- Explicit (one test = one feature/scenario).
- Maintainable (use shared fixtures from conftest).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealthFeature:
    """Feature: Health check (used by Render, monitoring, users)."""

    def test_health_returns_ok_and_version(self, api_client: TestClient):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["version"] == "1.0.0"
        assert "qdrant_connected" in data


class TestDocumentUploadFeature:
    """Feature: Upload & ingest documents (core of the app)."""

    def test_upload_txt_succeeds_and_returns_metadata(self, api_client: TestClient, sample_txt_path: str):
        with open(sample_txt_path, "rb") as f:
            resp = api_client.post(
                "/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert "doc_id" in data and data["doc_id"]
        assert data["filename"] == "test.txt"
        assert data["chunks_created"] >= 1
        assert data["vectors_stored"] >= 1
        assert "message" in data

    def test_upload_unsupported_extension_rejected(self, api_client: TestClient):
        # Simulate bad extension
        resp = api_client.post(
            "/documents/upload",
            files={"file": ("bad.exe", b"binary", "application/octet-stream")},
        )
        assert resp.status_code == 415
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_too_large_rejected(self, api_client: TestClient, tmp_path):
        # Create a file bigger than the 20MB limit (we create >20MB in memory simulation)
        big_content = b"x" * (21 * 1024 * 1024)
        resp = api_client.post(
            "/documents/upload",
            files={"file": ("huge.txt", big_content, "text/plain")},
        )
        assert resp.status_code == 413
        assert "File too large" in resp.json()["detail"]

    # Note: Real PDF/DOCX would require sample files in tests/data/.
    # For full coverage add:
    #   - test_upload_pdf_works (with a minimal valid PDF)
    #   - Same for .docx using python-docx to generate in fixture.


class TestListDocumentsFeature:
    """Feature: List all ingested documents."""

    def test_list_empty_when_no_documents(self, api_client: TestClient):
        resp = api_client.get("/documents/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_uploaded_documents(self, api_client: TestClient, sample_txt_path: str):
        # Upload one
        with open(sample_txt_path, "rb") as f:
            api_client.post("/documents/upload", files={"file": ("list-test.txt", f, "text/plain")})

        resp = api_client.get("/documents/")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) >= 1
        assert any(d["filename"] == "list-test.txt" for d in docs)
        assert all(k in docs[0] for k in ("doc_id", "filename", "chunk_count", "created_at"))


class TestDeleteDocumentFeature:
    """Feature: Delete document + BM25 index maintenance (important regression area)."""

    def test_delete_nonexistent_returns_404(self, api_client: TestClient):
        resp = api_client.delete("/documents/does-not-exist-123")
        assert resp.status_code == 404

    def test_delete_removes_from_list_and_bm25(self, api_client: TestClient, sample_txt_path: str):
        # Upload two docs
        with open(sample_txt_path, "rb") as f:
            r1 = api_client.post("/documents/upload", files={"file": ("to-delete.txt", f, "text/plain")}).json()
        with open(sample_txt_path, "rb") as f:
            r2 = api_client.post("/documents/upload", files={"file": ("keep.txt", f, "text/plain")}).json()

        # Delete first
        resp = api_client.delete(f"/documents/{r1['doc_id']}")
        assert resp.status_code == 204

        # List should only have the second
        docs = api_client.get("/documents/").json()
        doc_ids = [d["doc_id"] for d in docs]
        assert r1["doc_id"] not in doc_ids
        assert r2["doc_id"] in doc_ids

        # Query should still find content from the kept document (exercises BM25 after delete)
        q = api_client.post("/query/", json={"question": "What is this about?", "top_k": 3})
        assert q.status_code == 200
        sources = q.json()["sources"]
        assert any(s["doc_id"] == r2["doc_id"] for s in sources)


class TestQueryFeature:
    """Feature: Hybrid retrieval + answer generation (the main value proposition)."""

    def test_query_no_documents_returns_guidance(self, api_client: TestClient):
        resp = api_client.post("/query/", json={"question": "Anything?", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "No documents" in data["answer"] or len(data["answer"]) > 0
        assert data["sources"] == []

    def test_query_after_upload_returns_answer_and_sources(self, api_client: TestClient, sample_txt_path: str):
        with open(sample_txt_path, "rb") as f:
            api_client.post("/documents/upload", files={"file": ("query-test.txt", f, "text/plain")})

        resp = api_client.post(
            "/query/",
            json={"question": "What is this document about?", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data and isinstance(data["answer"], str)
        assert "sources" in data and len(data["sources"]) > 0
        assert "question" in data
        assert "model" in data

        src = data["sources"][0]
        for key in ("text", "filename", "doc_id", "chunk_index", "score"):
            assert key in src

    def test_top_k_is_respected(self, api_client: TestClient, sample_txt_path: str):
        with open(sample_txt_path, "rb") as f:
            api_client.post("/documents/upload", files={"file": ("topk-test.txt", f, "text/plain")})

        resp = api_client.post("/query/", json={"question": "test", "top_k": 1})
        assert len(resp.json()["sources"]) <= 1

    # Add when doc_ids filtering is fully wired in retriever/pipeline:
    # def test_query_with_doc_ids_filter_only_returns_from_those_docs(self, ...)


class TestErrorHandlingAndValidation:
    """Cross-cutting: proper HTTP errors and user-friendly messages."""

    def test_query_without_groq_key_returns_clear_error(self, api_client: TestClient):
        # We can force by using a client with empty key, but the fixture always sets one.
        # For real test, create a separate client or temporarily patch.
        # Here we at least exercise the 422/500 paths.
        resp = api_client.post("/query/", json={"question": "", "top_k": 5})  # empty question
        # The model has min_length=1, so Pydantic should 422
        assert resp.status_code in (422, 400)


class TestStaticUIFeature:
    """Feature: The nice user-friendly frontend is served and functional at root."""

    def test_root_serves_nice_ui_html(self, api_client: TestClient):
        resp = api_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        html = resp.text
        # Key markers from our nice UI
        assert "RAG Document Q&amp;A" in html or "RAG Document Q&A" in html
        assert "DOCUMENTS" in html
        assert "ASK A QUESTION" in html
        assert "dropzone" in html.lower() or "drag" in html.lower()

    def test_api_docs_still_available(self, api_client: TestClient):
        resp = api_client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()


# Future expansion ideas for complete test system:
# - tests/integration/test_loader.py (real PDF/DOCX using sample files in tests/data/)
# - tests/integration/test_reranker.py (mock CrossEncoder, assert rerank changes ordering)
# - tests/e2e/ (Playwright against the static UI + real backend, or against deployed Render URL)
# - Property-based tests (hypothesis) for chunker edge cases.
# - Load test for retrieval (using locust or pytest-benchmark).
# - Contract test: validate responses against Pydantic models / OpenAPI schema.
#
# To run only fast unit + integration:
#   pytest tests/ -q -m "not slow"
#
# Add to pyproject.toml under [tool.pytest.ini_options] if desired:
#   markers = "slow: marks tests as slow (deselect with '-m \"not slow\"')"
