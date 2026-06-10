"""Unit tests for HybridRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_qa.services.retriever import HybridRetriever, _rrf_score


# ---------------------------------------------------------------------------
# RRF formula tests
# ---------------------------------------------------------------------------


class TestRRFScore:
    def test_rank1_gives_max_score(self):
        score = _rrf_score(1)
        assert score == pytest.approx(1.0 / 61)

    def test_higher_rank_gives_lower_score(self):
        assert _rrf_score(1) > _rrf_score(10)

    def test_same_rank_same_score(self):
        assert _rrf_score(5) == pytest.approx(_rrf_score(5))

    def test_custom_k(self):
        assert _rrf_score(1, k=0) == pytest.approx(1.0)
        assert _rrf_score(1, k=10) == pytest.approx(1.0 / 11)


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


def make_mock_vector_store(hits=None):
    vs = MagicMock()
    vs.search.return_value = hits or []
    return vs


def make_mock_embedder(vec=None):
    emb = MagicMock()
    emb.embed_one.return_value = vec or [0.1] * 384
    return emb


SAMPLE_DOCS = [
    {"doc_id": "d1", "chunk_index": 0, "text": "the quick brown fox jumps", "filename": "a.txt"},
    {"doc_id": "d1", "chunk_index": 1, "text": "the lazy dog sleeps all day", "filename": "a.txt"},
    {"doc_id": "d2", "chunk_index": 0, "text": "machine learning models require data", "filename": "b.txt"},
]


class TestHybridRetrieverBM25:
    def test_empty_corpus_returns_empty(self):
        vs = make_mock_vector_store()
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        results = retriever._bm25_search("fox", top_n=5)
        assert results == []

    def test_bm25_finds_relevant_doc(self):
        vs = make_mock_vector_store()
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        texts = [d["text"] for d in SAMPLE_DOCS]
        retriever.add_documents(texts, SAMPLE_DOCS)

        results = retriever._bm25_search("fox", top_n=3)
        assert any("fox" in r["text"] for r in results)

    def test_add_documents_builds_index(self):
        vs = make_mock_vector_store()
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        texts = [d["text"] for d in SAMPLE_DOCS]
        retriever.add_documents(texts, SAMPLE_DOCS)
        assert len(retriever._corpus) == 3

    def test_clear_resets_corpus(self):
        vs = make_mock_vector_store()
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        texts = [d["text"] for d in SAMPLE_DOCS]
        retriever.add_documents(texts, SAMPLE_DOCS)
        retriever.clear()
        assert retriever._corpus == []


class TestHybridRetrieverFusion:
    def test_fusion_deduplicates(self):
        """Same doc appearing in both dense and BM25 should be fused (not duplicated)."""
        vs = make_mock_vector_store(hits=[SAMPLE_DOCS[0]])
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        texts = [d["text"] for d in SAMPLE_DOCS]
        retriever.add_documents(texts, SAMPLE_DOCS)

        results = retriever.search("fox", top_k=5)
        doc_keys = [f"{r['doc_id']}::{r['chunk_index']}" for r in results]
        assert len(doc_keys) == len(set(doc_keys)), "Duplicate results in fusion output"

    def test_search_respects_top_k(self):
        dense_hits = [
            {"doc_id": f"d{i}", "chunk_index": 0, "text": f"doc {i}", "filename": "x.txt"}
            for i in range(10)
        ]
        vs = make_mock_vector_store(hits=dense_hits)
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        results = retriever.search("query", top_k=3)
        assert len(results) <= 3

    def test_rrf_score_added_to_results(self):
        vs = make_mock_vector_store(hits=[SAMPLE_DOCS[0]])
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        results = retriever.search("fox", top_k=5)
        for r in results:
            assert "_rrf_score" in r

    def test_dense_only_search_works(self):
        """Without BM25 corpus, dense results still flow through."""
        dense_hits = [SAMPLE_DOCS[0], SAMPLE_DOCS[1]]
        vs = make_mock_vector_store(hits=dense_hits)
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        results = retriever.search("anything", top_k=5)
        assert len(results) >= 1
