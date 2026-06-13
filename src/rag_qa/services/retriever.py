"""Hybrid retriever: dense (cosine) + BM25 with RRF fusion."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_RRF_K = 60  # Standard RRF constant


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (k + rank)


class HybridRetriever:
    """Combine dense vector search and BM25 text search via RRF fusion."""

    def __init__(self, vector_store, embedder) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._corpus: list[str] = []       # raw texts for BM25
        self._corpus_meta: list[dict] = []  # corresponding payloads
        self._bm25 = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add_documents(self, texts: list[str], payloads: list[dict]) -> None:
        """Add texts + payloads to the BM25 index (append-only)."""
        self._corpus.extend(texts)
        self._corpus_meta.extend(payloads)
        self._bm25 = None  # invalidate cached index

    def clear(self) -> None:
        """Clear in-memory BM25 index."""
        self._corpus = []
        self._corpus_meta = []
        self._bm25 = None

    def remove_documents_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks belonging to the given doc_id from the BM25 corpus.
        Returns the number of chunks removed.
        """
        if not self._corpus:
            return 0

        original_len = len(self._corpus)
        kept_corpus: list[str] = []
        kept_meta: list[dict] = []
        for text, meta in zip(self._corpus, self._corpus_meta):
            if meta.get("doc_id") != doc_id:
                kept_corpus.append(text)
                kept_meta.append(meta)

        if len(kept_corpus) == original_len:
            return 0

        self._corpus = kept_corpus
        self._corpus_meta = kept_meta
        self._bm25 = None  # force rebuild on next search
        return original_len - len(self._corpus)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, top_n: int = 20) -> list[dict[str, Any]]:
        """
        Hybrid search returning top_k results after RRF fusion.

        Returns a list of payload dicts with added keys:
          _dense_rank, _bm25_rank, _rrf_score
        """
        dense_results = self._dense_search(query, top_n)
        bm25_results = self._bm25_search(query, top_n)
        return self._fuse(dense_results, bm25_results, top_k)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, top_n: int) -> list[dict]:
        """Return top_n results from Qdrant cosine search."""
        query_vec = self._embedder.embed_one(query)
        return self._vector_store.search(query_vec, top_k=top_n)

    def _bm25_search(self, query: str, top_n: int) -> list[dict]:
        """Return top_n results from BM25 sparse text search."""
        if not self._corpus:
            return []

        bm25 = self._get_bm25()
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get indices sorted by score descending
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[dict] = []
        for idx, score in indexed[:top_n]:
            if score > 0:
                payload = dict(self._corpus_meta[idx])
                payload["_score"] = float(score)
                results.append(payload)
        return results

    def _get_bm25(self):
        """Lazily build the BM25 index."""
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            tokenized = [text.lower().split() for text in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def _fuse(
        self,
        dense: list[dict],
        bm25: list[dict],
        top_k: int,
    ) -> list[dict]:
        """RRF fusion of dense and BM25 ranked lists."""
        # Build identity key: prefer (doc_id, chunk_index) if available
        def _key(item: dict) -> str:
            return f"{item.get('doc_id', '')}::{item.get('chunk_index', item.get('text', '')[:50])}"

        scores: dict[str, float] = {}
        items: dict[str, dict] = {}

        for rank, item in enumerate(dense, start=1):
            k = _key(item)
            scores[k] = scores.get(k, 0.0) + _rrf_score(rank)
            items[k] = item

        for rank, item in enumerate(bm25, start=1):
            k = _key(item)
            scores[k] = scores.get(k, 0.0) + _rrf_score(rank)
            if k not in items:
                items[k] = item

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        results: list[dict] = []
        for k in sorted_keys[:top_k]:
            entry = dict(items[k])
            entry["_rrf_score"] = scores[k]
            results.append(entry)

        return results
