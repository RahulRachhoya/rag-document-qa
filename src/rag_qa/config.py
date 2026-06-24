"""Pydantic-settings configuration for RAG Q&A."""

from __future__ import annotations

import os
from typing import Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # Embedding
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Qdrant storage. Precedence (most → least specific):
    #   1. qdrant_url set   -> remote Qdrant (Qdrant Cloud / self-hosted server)
    #   2. qdrant_path set  -> local on-disk embedded Qdrant (durable, no server,
    #                          ideal for free tiers with a persistent disk)
    #   3. neither           -> in-memory (":memory:"), ephemeral; dev/tests only
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = ""
    qdrant_collection: str = "rag_documents"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_top_n_rerank: int = 20

    # Memory optimization (for low-RAM / free tiers)
    low_memory: bool = False

    # Cross-encoder reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = False

    # CORS
    # Comma-separated list of allowed origins. If not set, defaults to [] for security.
    # Example: "https://example.com,https://app.example.com"
    cors_origins: list[str] = []

    # API
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def _apply_platform_defaults(self) -> Self:
        """Auto-enable low-memory mode on Render when LOW_MEMORY is not explicitly set."""
        if os.environ.get("LOW_MEMORY") is None and os.environ.get("RENDER", "").lower() == "true":
            object.__setattr__(self, "low_memory", True)

        if self.low_memory:
            if self.chunk_size > 256:
                object.__setattr__(self, "chunk_size", 256)
            object.__setattr__(self, "reranker_enabled", False)
            if self.embed_model in ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"):
                object.__setattr__(self, "embed_model", "BAAI/bge-small-en-v1.5")

        return self


settings = Settings()
