"""Integration-style tests for RAGPipeline with mocked LLM."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rag_qa.config import Settings
from rag_qa.pipeline import RAGPipeline


def make_test_settings(**kwargs) -> Settings:
    defaults = {
        "groq_api_key": "test-key",
        "qdrant_url": "",
        "reranker_enabled": False,
        "chunk_size": 200,
        "chunk_overlap": 20,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def write_temp_txt(content: str) -> str:
    """Write content to a temp .txt file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


@pytest.fixture()
def pipeline_with_mock_llm():
    """RAGPipeline with mocked LLM and Embedder (no heavy ML deps required)."""
    settings = make_test_settings()
    with (
        patch("rag_qa.pipeline.GroqLLM") as MockLLM,
        patch("rag_qa.pipeline.create_embedder") as MockEmbedder,
    ):
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "Mocked answer from the document."
        MockLLM.return_value = mock_llm_instance

        mock_embedder_instance = MagicMock()
        # embed() returns list of float vectors; embed_one() returns a single vector
        mock_embedder_instance.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
        mock_embedder_instance.embed_one.return_value = [0.1] * 384
        mock_embedder_instance.dimension = 384
        MockEmbedder.return_value = mock_embedder_instance

        p = RAGPipeline(settings)
        yield p


class TestRAGPipelineIngest:
    @pytest.mark.asyncio
    async def test_ingest_txt_returns_metadata(self, pipeline_with_mock_llm):
        path = write_temp_txt("This is a test document. " * 30)
        result = await pipeline_with_mock_llm.ingest(path, "test.txt")
        assert result["doc_id"] is not None
        assert result["filename"] == "test.txt"
        assert result["chunks"] >= 1
        assert result["vectors_stored"] >= 1

    @pytest.mark.asyncio
    async def test_ingest_registers_document(self, pipeline_with_mock_llm):
        path = write_temp_txt("Document content for registration test. " * 10)
        result = await pipeline_with_mock_llm.ingest(path, "register.txt")
        docs = pipeline_with_mock_llm.list_documents()
        doc_ids = [d["doc_id"] for d in docs]
        assert result["doc_id"] in doc_ids

    @pytest.mark.asyncio
    async def test_ingest_empty_file_raises(self, pipeline_with_mock_llm):
        path = write_temp_txt("   \n\t  ")
        with pytest.raises(ValueError):
            await pipeline_with_mock_llm.ingest(path, "empty.txt")

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file_raises(self, pipeline_with_mock_llm):
        with pytest.raises(FileNotFoundError):
            await pipeline_with_mock_llm.ingest("/nonexistent/file.txt", "nope.txt")


class TestRAGPipelineQuery:
    @pytest.mark.asyncio
    async def test_query_no_docs_returns_guidance(self, pipeline_with_mock_llm):
        # Fresh pipeline, no docs
        result = await pipeline_with_mock_llm.query("What is this about?")
        assert "answer" in result
        # Either a real answer or the "no docs" message
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @pytest.mark.asyncio
    async def test_query_after_ingest_calls_llm(self, pipeline_with_mock_llm):
        path = write_temp_txt("Machine learning is the study of algorithms. " * 20)
        await pipeline_with_mock_llm.ingest(path, "ml.txt")
        result = await pipeline_with_mock_llm.query("What is machine learning?")
        assert result["answer"] == "Mocked answer from the document."
        assert "sources" in result
        assert "question" in result

    @pytest.mark.asyncio
    async def test_query_returns_sources(self, pipeline_with_mock_llm):
        path = write_temp_txt("Python is a programming language. " * 20)
        await pipeline_with_mock_llm.ingest(path, "python.txt")
        result = await pipeline_with_mock_llm.query("What is Python?")
        assert isinstance(result["sources"], list)
        if result["sources"]:
            source = result["sources"][0]
            assert "text" in source
            assert "filename" in source

    @pytest.mark.asyncio
    async def test_delete_document_removes_it(self, pipeline_with_mock_llm):
        # Ingest two documents so we can prove the survivor still works after delete
        path_del = write_temp_txt("This document will be deleted. " * 10)
        del_result = await pipeline_with_mock_llm.ingest(path_del, "delete_me.txt")
        del_doc_id = del_result["doc_id"]

        path_keep = write_temp_txt("Python is a programming language used for many things. " * 10)
        keep_result = await pipeline_with_mock_llm.ingest(path_keep, "keep.txt")
        keep_doc_id = keep_result["doc_id"]

        # Delete the first one
        deleted = pipeline_with_mock_llm.delete_document(del_doc_id)
        assert deleted is True

        docs = pipeline_with_mock_llm.list_documents()
        assert not any(d["doc_id"] == del_doc_id for d in docs)
        assert any(d["doc_id"] == keep_doc_id for d in docs)

        # Query about the kept document — this exercises HybridRetriever after removal.
        # The remaining doc must still be retrievable (both dense + BM25 corpus for it must survive).
        result = await pipeline_with_mock_llm.query("What is Python?")
        assert result["answer"] == "Mocked answer from the document."
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0
        # The source(s) should come from the kept document, not the deleted one
        for s in result["sources"]:
            assert s["doc_id"] == keep_doc_id
            assert "Python" in s.get("text", "") or "programming" in s.get("text", "").lower()


class TestRAGPipelineWarmup:
    def test_warmup_loads_embedder(self, pipeline_with_mock_llm):
        """warmup() must touch the embedder so first request avoids cold-start cost."""
        pipeline_with_mock_llm.warmup()
        pipeline_with_mock_llm._embedder.embed_one.assert_called()

    def test_warmup_loads_reranker_when_enabled(self):
        """Regression: warmup() must pre-load the cross-encoder when reranking is enabled.

        Previously the cross-encoder lazy-loaded on the first query, causing a
        multi-second (observed ~28s) stall on the first user request after cold start.
        """
        settings = make_test_settings(reranker_enabled=True)
        with (
            patch("rag_qa.pipeline.GroqLLM") as MockLLM,
            patch("rag_qa.pipeline.create_embedder") as MockEmbedder,
            patch("rag_qa.pipeline.CrossEncoderReranker") as MockReranker,
        ):
            MockLLM.return_value = MagicMock()
            mock_embedder = MagicMock()
            mock_embedder.embed_one.return_value = [0.1] * 384
            mock_embedder.dimension = 384
            MockEmbedder.return_value = mock_embedder

            mock_reranker = MagicMock()
            MockReranker.return_value = mock_reranker

            p = RAGPipeline(settings)
            p.warmup()

            # The reranker must be exercised during warmup (triggers model load).
            mock_reranker.rerank.assert_called()

    def test_warmup_skips_reranker_when_disabled(self, pipeline_with_mock_llm):
        """When reranking is off, warmup() must not attempt to load a cross-encoder."""
        # Default fixture has reranker_enabled=False -> NoOpReranker.
        from rag_qa.services.reranker import NoOpReranker

        assert isinstance(pipeline_with_mock_llm._reranker, NoOpReranker)
        # Should not raise; NoOpReranker.rerank is a harmless pass-through.
        pipeline_with_mock_llm.warmup()
