"""Retriever - Retrieval with reranking"""
import logging
from typing import List, Optional, Dict, Any

from app.config import TOP_K, RERANK_TOP_K

logger = logging.getLogger(__name__)

class RetrievedChunk:
    """A retrieved chunk with score"""
    def __init__(self, content: str, metadata: dict, score: float = 0.0):
        self.page_content = content
        self.metadata = metadata
        self.score = score

class Retriever:
    """Retrieval pipeline with optional reranking"""
    
    def __init__(self, vector_store=None, top_k: int = TOP_K, rerank_top_k: int = RERANK_TOP_K):
        from app.services.vector_store import get_vector_store
        
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy load reranker model"""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                # Use a lighter reranker model
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Reranker model loaded")
            except ImportError:
                logger.warning("sentence-transformers not available, skipping reranking")
                self._reranker = False
        
        return self._reranker
    
    def retrieve(self, query: str, query_embedding: List[float],
                 document_ids: List[str] = None,
                 use_reranker: bool = True) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query"""
        
        # Initial retrieval
        filter_metadata = {"document_id": {"$in": document_ids}} if document_ids else None
        
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=self.top_k * 2 if use_reranker else self.top_k,
            filter_metadata=filter_metadata
        )
        
        if not results:
            return []
        
        # Rerank if enabled and reranker is available
        if use_reranker and self.reranker:
            results = self._rerank(query, results)
        
        # Convert to RetrievedChunk objects
        chunks = []
        for r in results[:self.top_k]:
            chunks.append(RetrievedChunk(
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0)
            ))
        
        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return chunks
    
    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not results:
            return results
        
        # Prepare query-document pairs
        pairs = [(query, r.get("content", "")) for r in results]
        
        # Get relevance scores
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for r, score in zip(results, scores):
            r["score"] = float(score)
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:self.rerank_top_k]
    
    def retrieve_with_context(self, query: str, query_embedding: List[float],
                              document_ids: List[str] = None) -> Dict[str, Any]:
        """Retrieve chunks with formatted context"""
        chunks = self.retrieve(query, query_embedding, document_ids)
        
        # Build context for LLM
        context = "\n\n---\n\n".join([
            f"[Source {i+1}]: {chunk.page_content}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Build citations
        citations = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page")
            section = chunk.metadata.get("section", "")
            
            citation = f"{source}"
            if page:
                citation += f" (page {page})"
            if section:
                citation += f" - {section}"
            
            citations.append({
                "text": chunk.page_content[:200] + "...",
                "source": citation,
                "score": chunk.score
            })
        
        return {
            "context": context,
            "chunks": chunks,
            "citations": citations,
            "num_chunks": len(chunks)
        }


def get_retriever() -> Retriever:
    """Get global retriever instance"""
    return Retriever()