"""Embedding Service - BGE-M3 via HuggingFace"""
import os
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Generate embeddings using BGE-M3 model"""
    
    def __init__(self, model_name: str = None, device: str = None):
        from app.config import EMBEDDING_MODEL, HF_TOKEN
        
        self.model_name = model_name or EMBEDDING_MODEL
        
        # Auto-detect device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self._model = None
        self._tokenizer = None
        self._hf_token = HF_TOKEN
    
    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            kwargs = {"device": self.device}
            if self._hf_token:
                kwargs["use_auth_token"] = self._hf_token
            
            logger.info(f"Loading BGE-M3 model on {self.device}...")
            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._tokenizer = self._model.tokenizer
            logger.info("BGE-M3 model loaded successfully")
        
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load the tokenizer"""
        if self._tokenizer is None:
            _ = self.model  # Trigger lazy load
        return self._tokenizer
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str], 
                        batch_size: int = 32,
                        show_progress: bool = False) -> List[List[float]]:
        """Embed multiple documents in batches"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def embed_with_metadata(self, chunks: List) -> List[dict]:
        """Embed chunks and return with metadata"""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embed_documents(texts)
        
        results = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            results.append({
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata
            })
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        test_emb = self.embed_query("test")
        return len(test_emb)


# Global instance
_embedding_service: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service