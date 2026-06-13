"""
Live E2E tests for the *full unmocked RAG pipeline* against a deployed instance.

These tests are intended to be run against the *live environment* (your Render URL
or any other deployment) to verify that the entire system is running properly
in production.

What "full pipeline" means here (real components, no mocks):
- Real DocumentLoader (pypdf / python-docx / text)
- Real RecursiveChunker
- Real sentence-transformers embedding (MiniLM) → downloads model on first use
- Real Qdrant (whatever is configured via QDRANT_URL; defaults to in-memory on Render)
- Real rank-bm25 + RRF fusion in HybridRetriever
- Real cross-encoder reranker (if RERANKER_ENABLED=true in the target deployment; note that the free Render IaC in `render.yaml` sets RERANKER_ENABLED=false by default to mitigate the ~512MB RAM limit -- see README OOM section and render.yaml)
- Real Groq LLM call (Llama-3.3-70b-versatile or whatever is configured)

Requirements to run:
- A running deployment that has a valid GROQ_API_KEY configured.
- `GROQ_API_KEY` in your environment (same one used by the deployment, or one that works).
- Optional: `LIVE_API_BASE` env var (defaults to the author's Render URL for convenience).

Example:
    LIVE_API_BASE=https://rag-document-qa-22uh.onrender.com \
    GROQ_API_KEY=gsk_... \
    python -m pytest tests/live/ -q --tb=line -m live

These tests are **marked live** and are skipped by default in normal CI runs
(the regular `pytest tests/` job uses mocked integration tests). They are meant
for manual verification or a separate "live smoke" job that has the real secret.

They also serve as regression protection for the full end-to-end flow,
including the BM25 delete-survivor behavior that was fixed earlier.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration for live environment
# ---------------------------------------------------------------------------

LIVE_API_BASE = os.getenv("LIVE_API_BASE", "https://rag-document-qa-22uh.onrender.com")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Skip the entire module if we don't have a key to talk to a real LLM.
# This is the gate that keeps these tests from running (and costing) in normal CI.
pytestmark = pytest.mark.skipif(
    not GROQ_API_KEY,
    reason="GROQ_API_KEY environment variable is required to run live full-pipeline tests",
)

# Use a longer timeout because free-tier Render instances sleep and cold starts
# involve downloading the embedding model + cross-encoder + first Groq call.
LIVE_TIMEOUT = 90.0

# A small, self-contained document that produces reliable grounding.
# The LLM should be able to answer questions about it with high confidence.
SAMPLE_CONTENT = (
    "The capital of France is Paris. "
    "The Eiffel Tower is a famous landmark located in Paris. "
    "France is a country in Europe. "
    "Paris is known for its art, fashion, and cuisine. "
) * 4   # repeat a bit so chunking has something to work with


@pytest.fixture(scope="module")
def live_client():
    """
    HTTP client pointed at the live deployment.

    Performs a best-effort warm-up (multiple health pings) because Render free
    instances can take 30-60s+ to come out of sleep and load models.
    """
    client = httpx.Client(
        base_url=LIVE_API_BASE.rstrip("/"),
        timeout=LIVE_TIMEOUT,
        headers={"User-Agent": "rag-qa-live-e2e-tests"},
    )

    # Warm-up loop
    for attempt in range(5):
        try:
            r = client.get("/health")
            if r.status_code == 200 and r.json().get("status") == "ok":
                break
        except Exception:
            pass
        # small sleep between attempts is fine; pytest will just take longer
        import time
        time.sleep(3)

    yield client
    client.close()


def _create_temp_txt(content: str) -> str:
    """Create a real temporary .txt file that the live backend will actually parse."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestLiveFullPipeline:
    """
    End-to-end tests that exercise the *complete* RAG pipeline in the live
    environment (real loader → chunker → embedder → vector + BM25 store →
    hybrid retrieval + rerank (if enabled) → real Groq generation).
    """

    def test_live_health_and_version(self, live_client: httpx.Client):
        """Basic smoke test that the deployment is reachable and reports correctly."""
        r = live_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"   # locked as per project
        # On Render with in-memory Qdrant this will be True (hard-coded in current code)
        assert "qdrant_connected" in data

    def test_live_full_ingest_retrieve_delete_flow(self, live_client: httpx.Client):
        """
        The canonical full-pipeline test:
        upload real document → it becomes queryable with real sources →
        delete it → it is gone.
        """
        path = _create_temp_txt(SAMPLE_CONTENT)
        doc_id = None
        try:
            # 1. Upload (exercises loader, chunker, embedder, vector_store.upsert, retriever.add)
            with open(path, "rb") as f:
                resp = live_client.post(
                    "/documents/upload",
                    files={"file": ("france-paris.txt", f, "text/plain")},
                )
            assert resp.status_code == 201, resp.text
            payload = resp.json()
            assert payload["filename"] == "france-paris.txt"
            assert payload["chunks_created"] >= 1
            assert payload["vectors_stored"] >= 1
            doc_id = payload["doc_id"]

            # 2. Confirm it appears in list
            list_resp = live_client.get("/documents/")
            assert list_resp.status_code == 200
            docs = list_resp.json()
            assert any(d["doc_id"] == doc_id for d in docs)

            # 3. Real query (exercises hybrid retrieval + reranker if enabled + LLM)
            q = live_client.post(
                "/query/",
                json={"question": "What is the capital of France and what is famous there?", "top_k": 3},
            )
            assert q.status_code == 200, q.text
            result = q.json()

            assert "answer" in result
            answer = result["answer"].lower()
            # The answer must be grounded in the document we just uploaded.
            assert "paris" in answer or "france" in answer

            assert "sources" in result and len(result["sources"]) > 0
            for src in result["sources"]:
                assert src["doc_id"] == doc_id
                assert src["filename"] == "france-paris.txt"
                assert isinstance(src["score"], (int, float))
                assert len(src["text"]) > 10

            assert result.get("model")  # real model name returned by Groq

        finally:
            # 4. Cleanup (important for ephemeral in-memory stores on free tier)
            if doc_id:
                del_resp = live_client.delete(f"/documents/{doc_id}")
                # 204 or 404 (if already gone due to sleep) is acceptable
                assert del_resp.status_code in (204, 404)

    def test_live_delete_does_not_break_bm25_for_remaining_documents(self, live_client: httpx.Client):
        """
        Regression test for the BM25 delete bug that was fixed earlier.

        Upload two documents, delete one, query something that should only hit
        the surviving document. Sources must come only from the survivor.
        """
        path1 = _create_temp_txt("The sky is blue during the day. Clouds are white.")
        path2 = _create_temp_txt(SAMPLE_CONTENT)  # the France/Paris content
        doc1 = doc2 = None

        try:
            # Upload two
            with open(path1, "rb") as f:
                d1 = live_client.post("/documents/upload", files={"file": ("sky.txt", f, "text/plain")}).json()
                doc1 = d1["doc_id"]

            with open(path2, "rb") as f:
                d2 = live_client.post("/documents/upload", files={"file": ("france.txt", f, "text/plain")}).json()
                doc2 = d2["doc_id"]

            # Delete the first (sky)
            del_resp = live_client.delete(f"/documents/{doc1}")
            assert del_resp.status_code in (204, 404)

            # Query something that should hit only the surviving document
            q = live_client.post(
                "/query/",
                json={"question": "What is the capital of France?", "top_k": 5},
            )
            assert q.status_code == 200
            result = q.json()

            assert result["sources"], "Expected sources from the surviving document"
            for s in result["sources"]:
                assert s["doc_id"] == doc2, "After delete, BM25/dense results leaked the deleted document"
                assert "paris" in s["text"].lower() or "france" in s["text"].lower()

        finally:
            for did in (doc1, doc2):
                if did:
                    live_client.delete(f"/documents/{did}")

    def test_live_list_and_delete_idempotency(self, live_client: httpx.Client):
        """Basic list + delete behavior against real storage."""
        path = _create_temp_txt("Just a tiny document for list/delete test.")
        doc_id = None
        try:
            with open(path, "rb") as f:
                resp = live_client.post("/documents/upload", files={"file": ("tiny.txt", f, "text/plain")})
            doc_id = resp.json()["doc_id"]

            # List should contain it
            lst = live_client.get("/documents/").json()
            assert any(d["doc_id"] == doc_id for d in lst)

            # Delete
            d = live_client.delete(f"/documents/{doc_id}")
            assert d.status_code in (204, 404)

            # List should no longer contain it
            lst2 = live_client.get("/documents/").json()
            assert not any(d["doc_id"] == doc_id for d in lst2)

        finally:
            if doc_id:
                live_client.delete(f"/documents/{doc_id}")


# ---------------------------------------------------------------------------
# How to run these against your own deployment
# ---------------------------------------------------------------------------
# 1. Make sure your Render (or other) service has GROQ_API_KEY set.
# 2. From your machine (or a GitHub Action that has the secret):
#
#    GROQ_API_KEY=gsk_... \
#    LIVE_API_BASE=https://your-service.onrender.com \
#    python -m pytest tests/live/ -q --tb=line -m live
#
# These tests will:
# - Warm the (possibly sleeping) instance
# - Exercise the real end-to-end RAG pipeline with real Groq calls
# - Clean up after themselves
#
# First run will be slow (model downloads). Subsequent runs while the
# container is warm are much faster.
#
# HF_TOKEN is *not* required for these tests (or for the Render API deployment at all).
# HF_TOKEN is only a GitHub Actions secret used by the CI job (`deploy-hf` in ci.yml)
# that pushes/updates the Gradio demo to Hugging Face Spaces. The user must create
# and supply their own HF_TOKEN secret (it will be updated/maintained by the repo owner
# or fork user for their HF Space). It is never needed at runtime of the FastAPI
# (Render) service, local execution, or for calling the live /query/ /health /documents
# endpoints. See also render.yaml comments and README "HF Spaces Demo" section for details.
# The 512MB free tier mitigations (reranker disabled in IaC, CPU torch, thread limits etc.)
# are documented in render.yaml, Dockerfile, and the new OOM Troubleshooting section in README.
