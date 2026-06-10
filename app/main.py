"""RAG Document Q&A - FastAPI Server"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy initialization of services
_embedding_service = None
_vector_store = None
_llm_service = None
_retriever = None

def get_embedding_service():
    """Get embedding service (lazy init)"""
    global _embedding_service
    if _embedding_service is None:
        from app.services.embedding import get_embedding_service as _get
        _embedding_service = _get()
    return _embedding_service

def get_vector_store():
    """Get vector store (lazy init)"""
    global _vector_store
    if _vector_store is None:
        from app.services.vector_store import get_vector_store as _get
        _vector_store = _get()
    return _vector_store

def get_llm_service():
    """Get LLM service (lazy init)"""
    global _llm_service
    if _llm_service is None:
        from app.services.llm import get_llm_service as _get
        _llm_service = _get()
    return _llm_service

def get_retriever():
    """Get retriever (lazy init)"""
    global _retriever
    if _retriever is None:
        from app.services.retriever import get_retriever as _get
        _retriever = _get()
    return _retriever


# ============= Pydantic Models =============

class QuestionRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    use_reranker: bool = True

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    citations: List[Dict[str, Any]]
    num_chunks: int

class DocumentInfo(BaseModel):
    document_id: str
    chunk_count: int
    sources: List[str]

class HealthResponse(BaseModel):
    status: str
    version: str


# ============= Lifespan =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting RAG Document Q&A server...")
    
    # Pre-initialize services
    try:
        get_embedding_service()
        get_vector_store()
        get_llm_service()
        logger.info("All services initialized")
    except Exception as e:
        logger.warning(f"Service initialization deferred: {e}")
    
    yield
    
    logger.info("Shutting down RAG Document Q&A server...")


# ============= App =============

app = FastAPI(
    title="RAG Document Q&A",
    description="Production RAG system with citation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============= Routes =============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve demo UI"""
    demo_path = STATIC_DIR / "demo.html"
    if demo_path.exists():
        return HTMLResponse(content=demo_path.read_text())
    
    return HTMLResponse(content="<h1>RAG Document Q&A API</h1><p>Visit /demo for the UI</p>")


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    """Serve demo UI"""
    demo_path = STATIC_DIR / "demo.html"
    if demo_path.exists():
        return HTMLResponse(content=demo_path.read_text())
    return HTMLResponse(content="<h1>Demo not found</h1>")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return HealthResponse(status="healthy", version="1.0.0")


# ============= Document Management =============

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload and process a document"""
    # Save uploaded file
    from app.config import UPLOADS_DIR
    
    file_path = UPLOADS_DIR / file.filename
    content = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    logger.info(f"Uploaded file: {file.filename}")
    
    # Process document
    try:
        from app.services.loader import DocumentLoader
        from app.services.chunker import TextChunker
        
        # Load
        documents = DocumentLoader.load_file(str(file_path))
        logger.info(f"Loaded {len(documents)} document sections")
        
        # Chunk
        chunker = TextChunker()
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Embed
        embed_service = get_embedding_service()
        embedded = embed_service.embed_with_metadata(chunks)
        logger.info(f"Embedded {len(embedded)} chunks")
        
        # Store
        vector_store = get_vector_store()
        doc_ids = vector_store.add_documents(embedded)
        
        return {
            "status": "success",
            "document_id": doc_ids[0] if doc_ids else None,
            "filename": file.filename,
            "chunk_count": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    try:
        vector_store = get_vector_store()
        doc_ids = vector_store.list_documents()
        
        docs = []
        for doc_id in doc_ids:
            chunks = vector_store.get_by_document_id(doc_id)
            sources = list(set(c.get("metadata", {}).get("source", "Unknown") for c in chunks))
            
            docs.append(DocumentInfo(
                document_id=doc_id,
                chunk_count=len(chunks),
                sources=sources
            ))
        
        return docs
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        vector_store = get_vector_store()
        count = vector_store.delete_by_document_id(document_id)
        
        return {
            "status": "success",
            "deleted_chunks": count
        }
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Query =============

@app.post("/query", response_model=QuestionResponse)
async def query_documents(request: QuestionRequest):
    """Query documents with a question"""
    try:
        # Get services
        embed_service = get_embedding_service()
        retriever = get_retriever()
        llm = get_llm_service()
        
        # Embed query
        query_embedding = embed_service.embed_query(request.question)
        
        # Retrieve relevant chunks
        retrieval_result = retriever.retrieve_with_context(
            query=request.question,
            query_embedding=query_embedding,
            document_ids=request.document_ids
        )
        
        if not retrieval_result["chunks"]:
            return QuestionResponse(
                answer="No relevant documents found. Please upload some documents first.",
                sources=[],
                citations=[],
                num_chunks=0
            )
        
        # Get answer from LLM
        response = llm.answer_question(
            question=request.question,
            context=retrieval_result["context"],
            citations=retrieval_result["citations"]
        )
        
        return QuestionResponse(
            answer=response.answer,
            sources=response.sources,
            citations=response.citations,
            num_chunks=retrieval_result["num_chunks"]
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def query_get(
    q: str = Query(..., description="Question to ask"),
    doc_id: Optional[str] = Query(None, description="Specific document ID")
):
    """GET endpoint for querying (for simple testing)"""
    doc_ids = [doc_id] if doc_id else None
    
    req = QuestionRequest(question=q, document_ids=doc_ids)
    return await query_documents(req)


# ============= Main =============

if __name__ == "__main__":
    import uvicorn
    from app.config import APP_HOST, APP_PORT
    
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)