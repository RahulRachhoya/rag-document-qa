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
        qdrant_path: str = "",
        embed_dim: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        self._client = None
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._qdrant_path = qdrant_path

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
            logger.info("Connecting to remote Qdrant at %s", self._qdrant_url)
            kwargs: dict[str, Any] = {"url": self._qdrant_url}
            if self._qdrant_api_key:
                kwargs["api_key"] = self._qdrant_api_key
            client = QdrantClient(**kwargs)
        elif self._qdrant_path:
            logger.info("Using local on-disk Qdrant at %s", self._qdrant_path)
            client = QdrantClient(path=self._qdrant_path)
        else:
            logger.info("Using in-memory Qdrant (ephemeral; data lost on restart)")
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

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """Dense cosine similarity search. Returns list of payload dicts with scores.

        When *doc_ids* is provided, results are restricted to those documents via a
        Qdrant payload filter (server-side, so recall is not lost to post-filtering).
        """
        query_filter = self._build_doc_filter(doc_ids)

        # qdrant-client >= 1.7 replaced .search() with .query_points()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        hits = response.points
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload["_score"] = hit.score
            results.append(payload)
        return results

    @staticmethod
    def _build_doc_filter(doc_ids: list[str] | None):
        """Build a Qdrant filter that restricts to the given doc_ids (or None)."""
        if not doc_ids:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        return Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))]
        )

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all vectors belonging to *doc_id*. Returns the number deleted."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        # Qdrant's delete response does not include a deleted count, so count the
        # matching points up front and report that (0 when nothing matched).
        deleted = self.client.count(
            collection_name=self.collection_name,
            count_filter=doc_filter,
            exact=True,
        ).count

        if deleted == 0:
            logger.info("No vectors found for doc_id=%s", doc_id)
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=doc_filter,
        )
        logger.info("Deleted %d vectors for doc_id=%s", deleted, doc_id)
        return deleted

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def _scroll_all_payloads(self):
        """Yield every stored point's payload dict, paging through the collection.

        Shared by list_documents() and iter_all_chunks() so the scroll/paging
        logic lives in one place.
        """
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                yield p.payload or {}
            if next_offset is None:
                break

    def list_documents(self) -> list[dict]:
        """Reconstruct the document registry from stored vector payloads.

        Qdrant is the source of truth: every chunk carries ``doc_id`` and
        ``filename`` in its payload, so the document list survives process
        restarts whenever a persistent Qdrant (``QDRANT_URL``/``QDRANT_PATH``)
        is configured.

        Returns one entry per ``doc_id``::

            {"doc_id": str, "filename": str, "chunk_count": int, "created_at": str | None}
        """
        agg: dict[str, dict] = {}
        for payload in self._scroll_all_payloads():
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            entry = agg.get(doc_id)
            if entry is None:
                agg[doc_id] = {
                    "doc_id": doc_id,
                    "filename": payload.get("filename", "unknown"),
                    "chunk_count": 1,
                    "created_at": payload.get("created_at"),
                }
            else:
                entry["chunk_count"] += 1
        return list(agg.values())

    def iter_all_chunks(self) -> list[dict]:
        """Return every stored chunk payload (text + metadata).

        Used to rehydrate the in-process BM25 index after a restart: the dense
        side lives in persistent Qdrant, and the sparse corpus is reconstructed
        from the same payloads so hybrid retrieval is not silently degraded to
        dense-only until documents are re-ingested.
        """
        return [p for p in self._scroll_all_payloads() if p.get("text")]
