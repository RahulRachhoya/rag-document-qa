"""Live end-to-end check that explain=True produces a fully-populated trace.

Mocks only the external LLM + embedder (same as the test suite); everything
else — chunking, vector store, BM25, RRF fusion, trace assembly — runs for real.
Asserts the exact data contract the UI's renderRetrievalViz() consumes.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_qa.config import Settings


def _write(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


async def main() -> None:
    with (
        patch("rag_qa.pipeline.GroqLLM") as MockLLM,
        patch("rag_qa.pipeline.create_embedder") as MockEmb,
    ):
        llm = MagicMock()
        llm.generate.return_value = "Gradient descent iteratively updates parameters to minimize loss."
        MockLLM.return_value = llm

        # Deterministic but query-sensitive embedding: bag-of-chars hash into 384 dims
        # so dense scores actually vary across chunks (not all-identical).
        def fake_embed(texts):
            out = []
            for t in texts:
                v = [0.0] * 384
                for ch in t.lower():
                    v[ord(ch) % 384] += 1.0
                norm = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / norm for x in v])
            return out

        emb = MagicMock()
        emb.embed.side_effect = fake_embed
        emb.embed_one.side_effect = lambda t: fake_embed([t])[0]
        emb.dimension = 384
        MockEmb.return_value = emb

        from rag_qa.pipeline import RAGPipeline

        settings = Settings(
            groq_api_key="test-key",
            qdrant_url="",
            reranker_enabled=False,
            chunk_size=200,
            chunk_overlap=20,
        )
        pipe = RAGPipeline(settings)

        doc = (
            "Gradient descent optimizes model parameters by following the negative gradient. "
            "The learning rate controls the size of each update step. "
            "Momentum accumulates past gradients to accelerate convergence. "
            "Adam combines momentum with per-parameter adaptive learning rates. "
            "Overfitting happens when a model memorizes training data instead of generalizing. "
            "Regularization techniques like dropout and weight decay reduce overfitting. "
        ) * 4
        path = _write(doc)
        try:
            await pipe.ingest(path, "ml.txt")
            result = await pipe.query("How does gradient descent update parameters?", explain=True)
        finally:
            Path(path).unlink(missing_ok=True)

        trace = result["trace"]
        assert trace is not None, "trace missing"

        problems: list[str] = []

        # Lane presence
        for key in ("question", "dense", "bm25", "fused", "reranked", "reranker_enabled", "timings_ms"):
            if key not in trace:
                problems.append(f"trace missing key: {key}")

        # Timings: the UI reads dense/bm25/fuse + total over all values
        timings = trace.get("timings_ms", {})
        for tk in ("dense", "bm25", "fuse"):
            if tk not in timings:
                problems.append(f"timings_ms missing per-lane key: {tk}")
        if not all(isinstance(v, (int, float)) and v >= 0 for v in timings.values()):
            problems.append(f"timings_ms has non-numeric/negative values: {timings}")

        # Dense lane fields the UI renders
        for c in trace.get("dense", []):
            for f in ("doc_id", "chunk_index", "filename", "rank", "score", "text_preview"):
                if f not in c:
                    problems.append(f"dense candidate missing field: {f}")
                    break

        # Fused lane fields
        for c in trace.get("fused", []):
            for f in ("rank", "rrf_score", "dense_rank", "bm25_rank", "text_preview"):
                if f not in c:
                    problems.append(f"fused candidate missing field: {f}")
                    break

        # Reranked lane fields (reranker disabled here -> reranker_enabled False)
        for c in trace.get("reranked", []):
            for f in ("rank", "rerank_score", "previous_rank", "text_preview"):
                if f not in c:
                    problems.append(f"reranked candidate missing field: {f}")
                    break

        # Ranks must be 1-indexed and contiguous in each lane
        for lane in ("dense", "bm25", "fused", "reranked"):
            ranks = [c["rank"] for c in trace.get(lane, [])]
            if ranks and ranks != list(range(1, len(ranks) + 1)):
                problems.append(f"{lane} ranks not 1..N contiguous: {ranks}")

        # reranker_enabled should reflect the NoOp reranker (disabled)
        if trace.get("reranker_enabled") is not False:
            problems.append(f"reranker_enabled expected False (NoOp), got {trace.get('reranker_enabled')}")

        print("=== TRACE SUMMARY ===")
        print(f"question        : {trace['question']!r}")
        print(f"dense count     : {len(trace['dense'])}")
        print(f"bm25 count      : {len(trace['bm25'])}")
        print(f"fused count     : {len(trace['fused'])}")
        print(f"reranked count  : {len(trace['reranked'])}")
        print(f"reranker_enabled: {trace['reranker_enabled']}")
        print(f"timings_ms      : {json.dumps(timings)}")
        if trace["fused"]:
            print(f"top fused       : {json.dumps(trace['fused'][0], indent=2)[:400]}")

        if problems:
            print("\n=== PROBLEMS ===")
            for p in problems:
                print(f"  - {p}")
            raise SystemExit(1)
        print("\nALL TRACE CONTRACT CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
