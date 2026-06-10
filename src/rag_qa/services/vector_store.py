"""Qdrant vector store wrapper (in-memory or cloud)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    """Manage document vectors in Qdrant (in-memory by default)."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        embed_dim: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        self._client = None
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key

    @property
    def client(self):
        """Lazy-init Qdrant client and ensure collection exists."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if self._qdrant_url:
            logger.info("Connecting to Qdrant at %s", self._qdrant_url)
            kwargs: dict[str, Any] = {"url": self._qdrant_url}
            if self._qdrant_api_key:
                kwargs["api_key"] = self._qdrant_api_key
            client = QdrantClient(**kwargs)
        else:
            logger.info("Using in-memory Qdrant")
            client = QdrantClient(":memory:")

        # Create collection if absent
        existing = [c.name for c in client.get_collections().collections]
        if self.collection_name not in existing:
            client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)

        return client

    def upsert(self, vectors: list[list[float]], payloads: list[dict]) -> int:
        """Insert or update vectors with associated payloads. Returns count."""
        from qdrant_client.models import PointStruct

        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=pay)
            for vec, pay in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Upserted %d vectors", len(points))
        return len(points)

    def search(self, query_vector: list[float], top_k: int = 20) -> list[dict]:
        """Dense cosine similarity search. Returns list of payload dicts with scores."""
        # qdrant-client >= 1.7 replaced .search() with .query_points()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )
        hits = response.points
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload["_score"] = hit.score
            results.append(payload)
        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all vectors belonging to *doc_id*. Returns deleted count."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info("Deleted vectors for doc_id=%s, result=%s", doc_id, result)
        return 1  # Qdrant async delete doesn't return count directly

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
