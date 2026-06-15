"""Unit tests for HybridRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock

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


class TestHybridRetrieverDocIdFilter:
    """The doc_ids filter must restrict BOTH the dense and BM25 sides."""

    def test_bm25_filters_to_requested_doc_ids(self):
        vs = make_mock_vector_store()
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        texts = [d["text"] for d in SAMPLE_DOCS]
        retriever.add_documents(texts, SAMPLE_DOCS)

        # "the" appears in both d1 chunks; restrict to d2 only -> no BM25 hits for "the"
        results = retriever._bm25_search("the", top_n=10, doc_ids=["d2"])
        assert all(r["doc_id"] == "d2" for r in results)

    def test_bm25_no_filter_returns_all_matches(self):
        vs = make_mock_vector_store()
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        texts = [d["text"] for d in SAMPLE_DOCS]
        retriever.add_documents(texts, SAMPLE_DOCS)

        results = retriever._bm25_search("the", top_n=10)
        # Both d1 chunks contain "the"; no filter -> both eligible
        assert any(r["doc_id"] == "d1" for r in results)

    def test_search_forwards_doc_ids_to_vector_store(self):
        """The dense side must pass doc_ids through to vector_store.search()."""
        vs = make_mock_vector_store(hits=[SAMPLE_DOCS[0]])
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        retriever.search("fox", top_k=5, doc_ids=["d1"])
        # Assert the dense store was called with the doc_ids kwarg
        _, kwargs = vs.search.call_args
        assert kwargs.get("doc_ids") == ["d1"]

    def test_search_default_doc_ids_is_none(self):
        vs = make_mock_vector_store(hits=[SAMPLE_DOCS[0]])
        emb = make_mock_embedder()
        retriever = HybridRetriever(vs, emb)
        retriever.search("fox", top_k=5)
        _, kwargs = vs.search.call_args
        assert kwargs.get("doc_ids") is None


class TestVectorStoreDeleteCount:
    """delete_by_doc_id must return the real matched count, never a hardcoded 1."""

    def _make_store(self, matched_count: int):
        from rag_qa.services.vector_store import VectorStore

        store = VectorStore.__new__(VectorStore)  # bypass __init__ (no live Qdrant)
        store.collection_name = "test"
        # `client` is a lazy property backed by `_client`; inject a mock there.
        store._client = MagicMock()
        store._client.count.return_value = MagicMock(count=matched_count)
        return store

    def test_returns_real_count_when_multiple_chunks(self):
        store = self._make_store(matched_count=7)
        assert store.delete_by_doc_id("d1") == 7
        store.client.delete.assert_called_once()

    def test_returns_zero_and_skips_delete_when_no_match(self):
        store = self._make_store(matched_count=0)
        assert store.delete_by_doc_id("missing") == 0
        store.client.delete.assert_not_called()
