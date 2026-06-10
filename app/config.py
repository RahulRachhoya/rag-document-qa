"""RAG Document Q&A - Configuration"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
UPLOADS_DIR = BASE_DIR / "uploads"
DOCUMENTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
VECTOR_TABLE = "document_chunks"
EMBEDDING_DIM = 1024  # BGE-M3 dimension

# AWS Bedrock
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
CLAUDE_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = "BAAI/bge-m3"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
RERANK_TOP_K = 3

# App
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")