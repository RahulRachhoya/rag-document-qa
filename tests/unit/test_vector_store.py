"""Unit tests for VectorStore — client selection precedence and on-disk durability."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_qa.services.vector_store import VectorStore

EMBED_DIM = 4


def _vec(seed: float) -> list[float]:
    return [seed, seed + 0.1, seed + 0.2, seed + 0.3]


# ---------------------------------------------------------------------------
# Client-selection precedence: url > path > in-memory
# ---------------------------------------------------------------------------


class TestClientSelection:
    def _patch_qdrant(self):
        """Patch QdrantClient and the collection bootstrap so no real I/O happens."""
        fake_client = MagicMock()
        fake_client.get_collections.return_value = MagicMock(collections=[])
        return fake_client

    def test_url_takes_precedence_over_path(self):
        fake = self._patch_qdrant()
        with patch("qdrant_client.QdrantClient", return_value=fake) as MockClient:
            store = VectorStore(
                qdrant_url="https://cloud.qdrant.example",
                qdrant_api_key="secret",
                qdrant_path="/tmp/should-be-ignored",
                embed_dim=EMBED_DIM,
            )
            _ = store.client
            # Remote path: called with url + api_key, never with path=
            _, kwargs = MockClient.call_args
            assert kwargs.get("url") == "https://cloud.qdrant.example"
            assert kwargs.get("api_key") == "secret"
            assert "path" not in kwargs

    def test_path_used_when_no_url(self):
        fake = self._patch_qdrant()
        with patch("qdrant_client.QdrantClient", return_value=fake) as MockClient:
            store = VectorStore(qdrant_path="/tmp/local-qdrant", embed_dim=EMBED_DIM)
            _ = store.client
            _, kwargs = MockClient.call_args
            assert kwargs.get("path") == "/tmp/local-qdrant"

    def test_memory_when_neither_url_nor_path(self):
        fake = self._patch_qdrant()
        with patch("qdrant_client.QdrantClient", return_value=fake) as MockClient:
            store = VectorStore(embed_dim=EMBED_DIM)
            _ = store.client
            args, kwargs = MockClient.call_args
            assert args == (":memory:",)


# ---------------------------------------------------------------------------
# Real on-disk durability: data must survive a new VectorStore instance
# pointing at the same path (simulates a process restart).
# ---------------------------------------------------------------------------


class TestOnDiskPersistence:
    def test_vectors_survive_a_restart(self, tmp_path):
        db_path = str(tmp_path / "qdrant_db")

        # First "process": ingest two chunks across two docs.
        store1 = VectorStore(qdrant_path=db_path, embed_dim=EMBED_DIM)
        store1.upsert(
            [_vec(0.0), _vec(0.5)],
            [
                {"doc_id": "doc-a", "filename": "a.txt", "text": "alpha", "created_at": "t0"},
                {"doc_id": "doc-b", "filename": "b.txt", "text": "beta", "created_at": "t0"},
            ],
        )
        assert store1.count() == 2
        # Release the on-disk lock before re-opening (embedded Qdrant is single-writer).
        del store1

        # Second "process": brand-new instance at the same path.
        store2 = VectorStore(qdrant_path=db_path, embed_dim=EMBED_DIM)
        assert store2.count() == 2
        docs = {d["doc_id"]: d for d in store2.list_documents()}
        assert set(docs) == {"doc-a", "doc-b"}
        assert docs["doc-a"]["filename"] == "a.txt"
        assert docs["doc-a"]["created_at"] == "t0"

    def test_list_documents_aggregates_chunks_per_doc(self, tmp_path):
        store = VectorStore(qdrant_path=str(tmp_path / "agg_db"), embed_dim=EMBED_DIM)
        store.upsert(
            [_vec(0.0), _vec(0.1), _vec(0.2)],
            [
                {"doc_id": "multi", "filename": "m.txt", "text": "one", "created_at": "t0"},
                {"doc_id": "multi", "filename": "m.txt", "text": "two", "created_at": "t0"},
                {"doc_id": "solo", "filename": "s.txt", "text": "x", "created_at": "t1"},
            ],
        )
        docs = {d["doc_id"]: d for d in store.list_documents()}
        assert docs["multi"]["chunk_count"] == 2
        assert docs["solo"]["chunk_count"] == 1
