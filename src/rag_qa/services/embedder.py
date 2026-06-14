"""Dense text embedders — Sentence-Transformers (default) or FastEmbed (low memory)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rag_qa.config import Settings

logger = logging.getLogger(__name__)


def _log_memory(label: str) -> None:
    try:
        import psutil

        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        logger.info("Memory %s: %.1f MB (RSS)", label, rss_mb)
    except Exception:
        pass


class Embedder(Protocol):
    """Common interface for dense embedding backends."""

    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerEmbedder:
    """sentence-transformers backend (higher quality, ~400+ MB RSS)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
            _log_memory("after SentenceTransformer load")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class FastEmbedder:
    """ONNX FastEmbed backend (~50–120 MB RSS, suitable for 512 MB hosts)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("Loading FastEmbed model: %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
            _log_memory("after FastEmbed load")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_one("dimension probe"))
        return self._dimension


def create_embedder(settings: Settings) -> Embedder:
    """Pick embedder backend from settings (FastEmbed when low_memory=True)."""
    if settings.low_memory:
        model = settings.embed_model
        # all-MiniLM-L6-v2 is not a FastEmbed catalog name; use equivalent 384-dim model.
        if model in ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"):
            model = "BAAI/bge-small-en-v1.5"
        return FastEmbedder(model)
    return SentenceTransformerEmbedder(settings.embed_model)


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "all-MiniLM-L6-v2", *, low_memory: bool = False) -> Embedder:
    """Return a cached singleton embedder (tests / scripts)."""
    if low_memory:
        return FastEmbedder(
            "BAAI/bge-small-en-v1.5"
            if model_name in ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2")
            else model_name
        )
    return SentenceTransformerEmbedder(model_name)