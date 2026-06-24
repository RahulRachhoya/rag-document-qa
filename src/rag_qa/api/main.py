"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None  # best effort for memory logging on constrained envs

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rag_qa import __version__
from rag_qa.api.routes import documents as documents_route
from rag_qa.api.routes import health, query
from rag_qa.config import Settings
from rag_qa.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = Settings()
pipeline = RAGPipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources on startup (replaces deprecated on_event)."""
    documents_route.set_pipeline(pipeline)
    if psutil:
        try:
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            logger.info(
                "Startup memory: %.1f MB (RSS) | reranker_enabled=%s | low_memory=%s | embed=%s",
                rss_mb,
                settings.reranker_enabled,
                settings.low_memory,
                settings.embed_model,
            )
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, pipeline.warmup)

    if psutil:
        try:
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            logger.info("Post-warmup memory: %.1f MB (RSS)", rss_mb)
        except Exception:
            pass

    logger.info("RAG pipeline ready (model=%s)", settings.groq_model)
    yield


app = FastAPI(
    title="RAG Document Q&A",
    version=__version__,
    description="Production-grade retrieval augmented generation API.",
    lifespan=lifespan,
)

# Use CORS origins from environment variable, default to empty list (secure by default)
# CORS_ORIGINS env var accepts comma-separated values, e.g.: "https://example.com,https://app.example.com"
allow_origins = settings.cors_origins if settings.cors_origins else []
if not allow_origins:
    logger.warning("CORS_ORIGINS not set; no cross-origin requests will be allowed.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(documents_route.router)
app.include_router(query.router)

# Serve the nice user-friendly UI (ui/index.html) at the root path.
# API routes (/health, /documents, /query, /docs) take precedence.
# This makes visiting the Render URL show the polished UI by default.
try:
    # Robust path discovery:
    # - /app/ui is the Docker/production path (WORKDIR /app + final COPY . . after pip install)
    # - Other candidates for local dev (src layout, cwd, etc.)
    candidates = [
        Path("/app/ui"),  # Production Docker layout (most important for Render)
        Path(__file__).resolve().parent.parent.parent.parent / "ui",  # src-layout dev
        Path.cwd() / "ui",  # running from project root
        Path(__file__).resolve().parents[3] / "ui",  # fallback
    ]
    
    ui_dir = None
    for candidate in candidates:
        if candidate.exists() and (candidate / "index.html").exists():
            ui_dir = candidate
            break
    
    if ui_dir:
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
        logger.info(f"Serving nice UI from {ui_dir}")
    else:
        logger.warning("ui/index.html not found in any candidate location — nice UI not mounted. Falling back to API only.")
except Exception as e:
    logger.warning(f"Could not mount nice UI: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_qa.api.main:app", host="0.0.0.0", port=8000, reload=True)
