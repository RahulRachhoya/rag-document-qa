"""Pydantic-settings configuration for RAG Q&A."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embedding
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Qdrant (in-memory when url is empty)
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_documents"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_top_n_rerank: int = 20

    # Cross-encoder reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True

    # API
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20


settings = Settings()
