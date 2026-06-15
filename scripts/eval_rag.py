"""RAG evaluation harness.

Measures retrieval quality (hit@k, MRR, recall@k) and per-stage latency
(embed / retrieve / rerank / generate) against a labeled dataset.

Retrieval metrics run fully offline (no LLM, no API cost). Generation
faithfulness is only evaluated when a GROQ_API_KEY is present.

Usage
-----
    python scripts/eval_rag.py                 # retrieval + latency only
    python scripts/eval_rag.py --with-llm      # also score generation faithfulness
    python scripts/eval_rag.py --top-k 5 --runs 3 --out eval/reports/baseline.json

The harness instruments latency by wrapping the pipeline's own service
objects, so it requires NO changes to production code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make `rag_qa` importable when run as a script from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_qa.config import Settings  # noqa: E402
from rag_qa.pipeline import RAGPipeline  # noqa: E402

EVAL_DIR = _REPO_ROOT / "eval"
CORPUS_DIR = EVAL_DIR / "corpus"
DATASET_PATH = EVAL_DIR / "dataset.json"


@dataclass
class StageTimer:
    """Accumulates per-stage timings (milliseconds) across queries."""

    embed: list[float] = field(default_factory=list)
    retrieve: list[float] = field(default_factory=list)
    rerank: list[float] = field(default_factory=list)
    generate: list[float] = field(default_factory=list)
    end_to_end: list[float] = field(default_factory=list)


class _TimedRetriever:
    """Wraps HybridRetriever.search to split embed vs. retrieve timing."""

    def __init__(self, retriever, timer: StageTimer) -> None:
        self._r = retriever
        self._timer = timer

    def search(self, query, top_k=5, top_n=20, doc_ids=None):
        # Time the dense-embed step separately from the search itself.
        t0 = time.perf_counter()
        _ = self._r._embedder.embed_one(query)
        t1 = time.perf_counter()
        results = self._r.search(query, top_k=top_k, top_n=top_n, doc_ids=doc_ids)
        t2 = time.perf_counter()
        self._timer.embed.append((t1 - t0) * 1000)
        self._timer.retrieve.append((t2 - t1) * 1000)
        return results

    def __getattr__(self, name):
        return getattr(self._r, name)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "p50": round(_pct(values, 50), 2),
        "p95": round(_pct(values, 95), 2),
        "p99": round(_pct(values, 99), 2),
        "max": round(max(values), 2),
    }


def _chunk_is_hit(chunk: dict, item: dict) -> bool:
    """A chunk is a hit if it comes from the expected doc AND contains a relevant term."""
    text = (chunk.get("text") or "").lower()
    filename = (chunk.get("filename") or "").lower()
    expected = item["expected_doc"].lower()
    if expected not in filename:
        return False
    return any(term.lower() in text for term in item["relevant_terms"])


def _retrieval_metrics(ranked: list[dict], item: dict, top_k: int) -> dict[str, float]:
    """Compute hit@k, reciprocal rank, and recall for one query."""
    hit_rank = None
    hits = 0
    for rank, chunk in enumerate(ranked[:top_k], start=1):
        if _chunk_is_hit(chunk, item):
            hits += 1
            if hit_rank is None:
                hit_rank = rank
    return {
        "hit": 1.0 if hit_rank else 0.0,
        "reciprocal_rank": (1.0 / hit_rank) if hit_rank else 0.0,
        "hits_in_topk": hits,
    }


def ingest_corpus(pipeline: RAGPipeline) -> int:
    """Ingest every file in the eval corpus. Returns total chunks."""
    import asyncio

    total = 0
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        res = asyncio.run(pipeline.ingest(str(path), path.name))
        total += res["chunks"]
        print(f"  ingested {path.name}: {res['chunks']} chunks")
    return total


def run_eval(top_k: int, runs: int, with_llm: bool) -> dict[str, Any]:
    dataset = json.loads(DATASET_PATH.read_text())
    items = dataset["items"]

    settings = Settings()
    has_groq = bool(settings.groq_api_key)
    do_llm = with_llm and has_groq

    print(f"Building pipeline (low_memory={settings.low_memory}, "
          f"reranker={settings.reranker_enabled}, llm={'on' if do_llm else 'off'})")
    pipeline = RAGPipeline(settings)
    pipeline.warmup()

    print("Ingesting eval corpus...")
    total_chunks = ingest_corpus(pipeline)

    timer = StageTimer()
    timed_retriever = _TimedRetriever(pipeline._retriever, timer)

    per_query: list[dict] = []
    faithful_hits = 0
    faithful_total = 0

    print(f"Running {len(items)} queries x {runs} run(s)...")
    for item in items:
        # Warm + measured runs; keep the last run's ranked list for quality scoring.
        ranked: list[dict] = []
        for _ in range(runs):
            e2e0 = time.perf_counter()
            ranked = timed_retriever.search(item["question"], top_k=top_k, top_n=top_k)
            t_rerank0 = time.perf_counter()
            ranked = pipeline._reranker.rerank(item["question"], ranked, top_k=top_k)
            t_rerank1 = time.perf_counter()
            timer.rerank.append((t_rerank1 - t_rerank0) * 1000)

            if do_llm:
                t_gen0 = time.perf_counter()
                answer = pipeline._llm.generate(item["question"], ranked)
                t_gen1 = time.perf_counter()
                timer.generate.append((t_gen1 - t_gen0) * 1000)
            timer.end_to_end.append((time.perf_counter() - e2e0) * 1000)

        metrics = _retrieval_metrics(ranked, item, top_k)

        if do_llm:
            answer = pipeline._llm.generate(item["question"], ranked)
            faithful_total += 1
            low = answer.lower()
            ok = all(s.lower() in low for s in item["answer_must_contain"])
            faithful_hits += 1 if ok else 0
            metrics["answer_faithful"] = 1.0 if ok else 0.0

        per_query.append({"id": item["id"], "question": item["question"], **metrics})

    n = len(items)
    report = {
        "config": {
            "top_k": top_k,
            "runs": runs,
            "llm_scored": do_llm,
            "low_memory": settings.low_memory,
            "reranker_enabled": settings.reranker_enabled,
            "embed_model": settings.embed_model,
            "corpus_chunks": total_chunks,
            "num_queries": n,
        },
        "retrieval": {
            "hit@k": round(sum(q["hit"] for q in per_query) / n, 4),
            "mrr": round(sum(q["reciprocal_rank"] for q in per_query) / n, 4),
            "avg_hits_in_topk": round(sum(q["hits_in_topk"] for q in per_query) / n, 2),
        },
        "latency_ms": {
            "embed": _summarize(timer.embed),
            "retrieve": _summarize(timer.retrieve),
            "rerank": _summarize(timer.rerank),
            "generate": _summarize(timer.generate),
            "end_to_end": _summarize(timer.end_to_end),
        },
        "per_query": per_query,
    }
    if do_llm:
        report["generation"] = {
            "faithfulness": round(faithful_hits / faithful_total, 4) if faithful_total else 0.0,
            "scored": faithful_total,
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG eval harness")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3, help="measured runs per query for latency")
    ap.add_argument("--with-llm", action="store_true", help="score generation faithfulness")
    ap.add_argument("--out", type=str, default="eval/reports/latest.json")
    args = ap.parse_args()

    report = run_eval(top_k=args.top_k, runs=args.runs, with_llm=args.with_llm)

    out_path = _REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    r = report["retrieval"]
    lat = report["latency_ms"]
    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY")
    print(f"  hit@{args.top_k}:          {r['hit@k']:.4f}")
    print(f"  MRR:             {r['mrr']:.4f}")
    print(f"  avg hits in top-k: {r['avg_hits_in_topk']}")
    print("\nLATENCY (ms)        p50      p95      p99      max")
    for stage in ("embed", "retrieve", "rerank", "generate", "end_to_end"):
        s = lat[stage]
        if s["count"]:
            print(f"  {stage:<12} {s['p50']:>8} {s['p95']:>8} {s['p99']:>8} {s['max']:>8}")
    if "generation" in report:
        print(f"\nGENERATION faithfulness: {report['generation']['faithfulness']:.4f}")
    print("=" * 60)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
