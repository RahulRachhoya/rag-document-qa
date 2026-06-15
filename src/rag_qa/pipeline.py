"""RAGPipeline: orchestrates ingest and query flows."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime

from rag_qa.config import Settings
from rag_qa.services.chunker import RecursiveChunker
from rag_qa.services.embedder import Embedder, create_embedder
from rag_qa.services.llm import GroqLLM
from rag_qa.services.loader import DocumentLoader
from rag_qa.services.reranker import CrossEncoderReranker, NoOpReranker
from rag_qa.services.retriever import HybridRetriever
from rag_qa.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage::

        pipeline = RAGPipeline(settings)
        await pipeline.ingest("path/to/doc.pdf", "doc.pdf")
        result = await pipeline.query("What is the main topic?")
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._loader = DocumentLoader()
        self._chunker = RecursiveChunker(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        self._embedder: Embedder = create_embedder(self._settings)
        self._vector_store = VectorStore(
            collection_name=self._settings.qdrant_collection,
            qdrant_url=self._settings.qdrant_url,
            qdrant_api_key=self._settings.qdrant_api_key,
            qdrant_path=self._settings.qdrant_path,
            embed_dim=self._settings.embed_dim,
        )
        self._retriever = HybridRetriever(self._vector_store, self._embedder)
        self._reranker: CrossEncoderReranker | NoOpReranker = (
            CrossEncoderReranker(self._settings.reranker_model)
            if self._settings.reranker_enabled
            else NoOpReranker()
        )
        self._llm = GroqLLM(
            api_key=self._settings.groq_api_key,
            model=self._settings.groq_model,
        )
        # In-memory doc registry (doc_id -> metadata)
        self._docs: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def ingest(self, file_path: str, filename: str) -> dict:
        """
        Load, chunk, embed, and store a document.

        Returns::

            {"doc_id": str, "chunks": int, "vectors_stored": int, "filename": str}
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._ingest_sync, file_path, filename)

    async def query(
        self, question: str, top_k: int = 5, doc_ids: list[str] | None = None, explain: bool = False
    ) -> dict:
        """
        Retrieve relevant chunks and generate a grounded answer.

        When *doc_ids* is provided, retrieval is restricted to those documents.
        When *explain* is True, the result carries a ``trace`` dict describing each
        retrieval stage (dense, bm25, fused, reranked) for visualization.

        Returns::

            {"answer": str, "sources": list[dict], "scores": list[float], "question": str}
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._query_sync, question, top_k, doc_ids, explain
        )

    def warmup(self) -> None:
        """Pre-load models at startup (avoids cold-start OOM/timeouts on first request).

        Loads the embedder always, and the cross-encoder reranker when enabled, so the
        first user query does not pay the model-load + first-inference cost (which can be
        tens of seconds for the cross-encoder).
        """
        backend = "FastEmbed" if self._settings.low_memory else "SentenceTransformer"
        logger.info("Warming up embedder (%s, model=%s)", backend, self._settings.embed_model)
        self._embedder.embed_one("warmup")
        logger.info("Embedder warmup complete (dim=%d)", self._embedder.dimension)

        if self._settings.reranker_enabled and not isinstance(self._reranker, NoOpReranker):
            logger.info("Warming up reranker (model=%s)", self._settings.reranker_model)
            self._reranker.rerank("warmup", [{"text": "warmup"}], top_k=1)
            logger.info("Reranker warmup complete")

    def list_documents(self) -> list[dict]:
        """Return metadata for all ingested documents.

        The in-memory cache is authoritative within a process lifetime. After a
        restart the cache is empty, so we reconstruct the registry from Qdrant
        payloads (the persistent source of truth when QDRANT_URL is set) and
        repopulate the cache. With an in-memory Qdrant this still correctly
        returns an empty list after restart, matching reality.
        """
        if self._docs:
            return list(self._docs.values())

        try:
            rebuilt = self._vector_store.list_documents()
        except Exception:  # pragma: no cover - defensive: never fail the list endpoint
            logger.exception("Failed to reconstruct document registry from vector store")
            return []

        for entry in rebuilt:
            self._docs[entry["doc_id"]] = entry
        return list(self._docs.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its vectors from the store.

        Works for documents ingested in the current process (cache hit) and for
        documents that survive in a persistent Qdrant across a restart (the
        cache is empty but the vectors still exist).
        """
        known = doc_id in self._docs
        if not known:
            # Cache miss: the doc may still live in a persistent Qdrant. Rebuild
            # the registry once and re-check before reporting "not found".
            self.list_documents()
            known = doc_id in self._docs

        if not known:
            return False

        self._vector_store.delete_by_doc_id(doc_id)
        self._docs.pop(doc_id, None)
        # Remove the deleted doc's chunks from the BM25 index (dense side already cleaned in Qdrant)
        self._retriever.remove_documents_by_doc_id(doc_id)
        return True

    # ------------------------------------------------------------------
    # Synchronous implementations (run in executor)
    # ------------------------------------------------------------------

    def _ingest_sync(self, file_path: str, filename: str) -> dict:
        doc_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        logger.info("Ingesting %s (doc_id=%s)", filename, doc_id)

        # 1. Load
        text = self._loader.load(file_path)
        if not text.strip():
            raise ValueError(f"No text extracted from {filename}")

        # 2. Chunk
        metadata_base = {"doc_id": doc_id, "filename": filename}
        chunks = self._chunker.split(text, metadata=metadata_base)
        if not chunks:
            raise ValueError(f"No chunks created from {filename}")

        # 3. Embed
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed(texts)

        # 4. Build payloads (created_at is stored on every chunk so the document
        #    registry can be reconstructed from Qdrant after a process restart)
        payloads = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": c.index,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "created_at": created_at,
            }
            for c in chunks
        ]

        # 5. Store in Qdrant
        stored = self._vector_store.upsert(vectors, payloads)

        # 6. Update BM25 index
        self._retriever.add_documents(texts, payloads)

        # 7. Register doc
        self._docs[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(chunks),
            "created_at": created_at,
        }

        logger.info("Ingested %s: %d chunks, %d vectors", filename, len(chunks), stored)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "vectors_stored": stored,
        }

    def _query_sync(
        self,
        question: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        explain: bool = False,
    ) -> dict:
        logger.info("Query: %.80s", question)
        if getattr(self._settings, "low_memory", False) or not self._settings.reranker_enabled:
            top_n = self._settings.retrieval_top_k
        else:
            top_n = self._settings.retrieval_top_n_rerank

        timings: dict[str, float] = {}
        # Only allocate a trace sink when explaining; the hot path stays untouched.
        retrieval_trace: dict[str, list[dict]] | None = {} if explain else None

        # 1. Hybrid retrieval
        t0 = time.perf_counter()
        candidates = self._retriever.search(
            question, top_k=top_n, top_n=top_n, doc_ids=doc_ids, trace=retrieval_trace
        )
        if explain:
            retrieve_ms = (time.perf_counter() - t0) * 1000.0
            # Prefer the retriever's per-lane sub-timings (dense/bm25/fuse); fall
            # back to a single coarse "retrieve" bucket if they weren't reported.
            sub = (retrieval_trace or {}).get("timings") if retrieval_trace else None
            if isinstance(sub, dict):
                timings.update({k: float(v) for k, v in sub.items()})
            else:
                timings["retrieve"] = retrieve_ms
        if not candidates:
            return {
                "answer": "No documents have been ingested yet. Please upload a document first.",
                "sources": [],
                "scores": [],
                "question": question,
                "trace": self._build_trace(question, retrieval_trace, [], [], timings)
                if explain
                else None,
            }

        # 2. Rerank
        t0 = time.perf_counter()
        reranked = self._reranker.rerank(question, candidates, top_k=top_k)
        if explain:
            timings["rerank"] = (time.perf_counter() - t0) * 1000.0

        # 3. Generate answer
        t0 = time.perf_counter()
        answer = self._llm.generate(question, reranked)
        if explain:
            timings["generate"] = (time.perf_counter() - t0) * 1000.0

        sources = [
            {
                "text": r.get("text", ""),
                "filename": r.get("filename", "unknown"),
                "doc_id": r.get("doc_id", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": r.get("_rerank_score", r.get("_rrf_score", 0.0)),
            }
            for r in reranked
        ]
        scores = [s["score"] for s in sources]

        return {
            "answer": answer,
            "sources": sources,
            "scores": scores,
            "question": question,
            "model": self._settings.groq_model,
            "trace": self._build_trace(question, retrieval_trace, candidates, reranked, timings)
            if explain
            else None,
        }

    # ------------------------------------------------------------------
    # Trace assembly (explain mode only)
    # ------------------------------------------------------------------

    @staticmethod
    def _preview(text: str, limit: int = 160) -> str:
        text = (text or "").strip().replace("\n", " ")
        return text[:limit] + ("\u2026" if len(text) > limit else "")

    def _build_trace(
        self,
        question: str,
        retrieval_trace: dict[str, list[dict]] | None,
        fused: list[dict],
        reranked: list[dict],
        timings: dict[str, float],
    ) -> dict:
        """Assemble the per-stage retrieval trace from captured intermediate data.

        Pure data shaping over lists the pipeline already produced — nothing is
        recomputed, so this is cheap and only runs when explain=True.
        """
        rt = retrieval_trace or {}

        def _candidate(item: dict, rank: int, score_key: str) -> dict:
            return {
                "doc_id": item.get("doc_id", ""),
                "chunk_index": item.get("chunk_index", 0),
                "filename": item.get("filename", "unknown"),
                "rank": rank,
                "score": float(item.get(score_key, 0.0)),
                "text_preview": self._preview(item.get("text", "")),
            }

        dense = [
            _candidate(it, i, "_score") for i, it in enumerate(rt.get("dense", []), start=1)
        ]
        bm25 = [
            _candidate(it, i, "_score") for i, it in enumerate(rt.get("bm25", []), start=1)
        ]

        fused_trace = [
            {
                "doc_id": it.get("doc_id", ""),
                "chunk_index": it.get("chunk_index", 0),
                "filename": it.get("filename", "unknown"),
                "dense_rank": it.get("_dense_rank"),
                "bm25_rank": it.get("_bm25_rank"),
                "rrf_score": float(it.get("_rrf_score", 0.0)),
                "rank": i,
                "text_preview": self._preview(it.get("text", "")),
            }
            for i, it in enumerate(fused, start=1)
        ]

        # Position of each fused chunk so we can show how far rerank moved it.
        def _key(it: dict) -> str:
            return f"{it.get('doc_id', '')}::{it.get('chunk_index', 0)}"

        fused_pos = {_key(it): i for i, it in enumerate(fused, start=1)}
        reranked_trace = [
            {
                "doc_id": it.get("doc_id", ""),
                "chunk_index": it.get("chunk_index", 0),
                "filename": it.get("filename", "unknown"),
                "rerank_score": float(it.get("_rerank_score", it.get("_rrf_score", 0.0))),
                "rank": i,
                "previous_rank": fused_pos.get(_key(it)),
                "text_preview": self._preview(it.get("text", "")),
            }
            for i, it in enumerate(reranked, start=1)
        ]

        return {
            "question": question,
            "dense": dense,
            "bm25": bm25,
            "fused": fused_trace,
            "reranked": reranked_trace,
            "reranker_enabled": not isinstance(self._reranker, NoOpReranker),
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }
