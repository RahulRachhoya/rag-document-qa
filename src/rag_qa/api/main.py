"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from pathlib import Path

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

app = FastAPI(
    title="RAG Document Q&A",
    version=__version__,
    description="Production-grade retrieval augmented generation API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise shared resources on startup."""
    documents_route.set_pipeline(pipeline)
    logger.info("RAG pipeline ready (model=%s)", settings.groq_model)


app.include_router(health.router)
app.include_router(documents_route.router)
app.include_router(query.router)

# Serve the nice user-friendly UI (ui/index.html) at the root path.
# API routes (/health, /documents, /query, /docs) take precedence.
# This makes visiting the Render URL show the polished UI by default.
try:
    # Calculate ui/ relative to the installed package layout (works in Docker + local dev)
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    ui_dir = base_dir / "ui"
    
    if ui_dir.exists() and (ui_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
        logger.info(f"Serving nice UI from {ui_dir}")
    else:
        logger.warning("ui/index.html not found — nice UI not mounted. Falling back to API only.")
except Exception as e:
    logger.warning(f"Could not mount nice UI: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_qa.api.main:app", host="0.0.0.0", port=8000, reload=True)
