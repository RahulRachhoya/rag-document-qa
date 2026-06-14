"""Cross-encoder reranker for retrieved chunks."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device="cpu")
            try:
                import os

                import psutil

                rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                logger.info("Memory after reranker load: %.1f MB (RSS)", rss_mb)
            except Exception:
                pass
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Rerank *chunks* by relevance to *query*.

        Each chunk dict must have a 'text' key.
        Returns top_k chunks sorted by descending cross-encoder score,
        with '_rerank_score' added.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        results: list[dict] = []
        for score, chunk in scored[:top_k]:
            entry = dict(chunk)
            entry["_rerank_score"] = float(score)
            results.append(entry)

        logger.debug("Reranked %d -> %d chunks", len(chunks), len(results))
        return results


class NoOpReranker:
    """Pass-through reranker (disabled state)."""

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """Return the top_k chunks unchanged."""
        return chunks[:top_k]
