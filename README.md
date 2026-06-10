# RAG Document Q&A

[![CI](https://github.com/RahulRachhoya/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/RahulRachhoya/rag-document-qa/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-orange.svg)](https://huggingface.co/spaces/RahulRachhoya/rag-document-qa)

Production-grade Retrieval Augmented Generation (RAG) system that answers questions from uploaded documents using hybrid search and Groq LLM.

## Features

- **Hybrid retrieval**: dense cosine search (Qdrant) + BM25 sparse search fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking**: top-20 candidates reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Groq LLM**: Llama-3.3-70b-versatile for fast, free inference
- **Multi-format ingestion**: PDF, DOCX, TXT, Markdown
- **In-memory Qdrant**: zero config for demo; pluggable for cloud Qdrant
- **FastAPI backend**: REST endpoints for upload, list, delete, query
- **Gradio demo**: hosted on Hugging Face Spaces

## Architecture

```
PDF / DOCX / TXT
       |
       v
  [DocumentLoader]
   pypdf / python-docx
       |
       v
  [RecursiveChunker]
   512 tokens / 64 overlap
       |
       v
  [MiniLM Embedder]
   all-MiniLM-L6-v2
      / \
     /   \
    v     v
[Qdrant]  [BM25]
 cosine   Okapi
    \     /
     \   /
      v v
  [RRF Fusion]
  1/(k + rank)
       |
       v
[Cross-Encoder Rerank]
 ms-marco-MiniLM-L-6-v2
       |
       v
  [Groq LLM]
  Llama-3.3-70b-versatile
       |
       v
    Answer
```

## Quick Start

```bash
git clone https://github.com/RahulRachhoya/rag-document-qa.git
cd rag-document-qa

# Install
pip install -e ".[dev]"

# Set your Groq key (free at console.groq.com)
echo "GROQ_API_KEY=gsk_..." > .env

# Run the API
uvicorn rag_qa.api.main:app --reload

# Or run the Gradio demo locally
python hf_demo/app.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload and ingest a document |
| GET | `/documents/` | List all documents |
| DELETE | `/documents/{doc_id}` | Delete a document |
| POST | `/query/` | Ask a question |

### Example

```bash
# Upload a document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@myreport.pdf"

# Ask a question
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5}'
```

## Project Layout

```
src/rag_qa/
  config.py          pydantic-settings BaseSettings
  models.py          Pydantic request/response models
  pipeline.py        RAGPipeline (ingest + query)
  services/
    loader.py        PDF/DOCX/TXT document loading
    chunker.py       Recursive character text splitter
    embedder.py      SentenceTransformers all-MiniLM-L6-v2
    vector_store.py  Qdrant client (in-memory or cloud)
    retriever.py     Hybrid dense + BM25 + RRF
    reranker.py      Cross-encoder reranking
    llm.py           Groq Llama chat completion
  api/
    main.py          FastAPI app
    routes/
      health.py
      documents.py
      query.py
tests/unit/          Mock-based pytest suite (no API keys needed)
hf_demo/app.py       Gradio two-tab demo for HF Spaces
```

## Running Tests

```bash
pytest tests/unit -v
```

Tests use mocked LLM calls and in-memory Qdrant -- no API keys required.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Groq API key from console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `QDRANT_URL` | `""` (in-memory) | Qdrant cloud URL |
| `QDRANT_API_KEY` | `""` | Qdrant cloud API key |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |

## HF Spaces Demo

Live demo: [huggingface.co/spaces/RahulRachhoya/rag-document-qa](https://huggingface.co/spaces/RahulRachhoya/rag-document-qa)

To deploy your own:
1. Fork this repo
2. Add `HF_TOKEN` as a GitHub Actions secret
3. Push to `main` -- CI will deploy automatically

## License

MIT
