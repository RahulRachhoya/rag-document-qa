"""Vector Store - Supabase pgvector integration"""
import os
import logging
from typing import List, Optional, Dict, Any
import uuid

import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    """Supabase pgvector vector store"""
    
    def __init__(self, supabase_client=None):
        from app.config import SUPABASE_URL, SUPABASE_ANON_KEY
        from app.config import VECTOR_TABLE, EMBEDDING_DIM
        
        self.table_name = VECTOR_TABLE
        self.embedding_dim = EMBEDDING_DIM
        self._client = supabase_client
        
        # Initialize Supabase client if not provided
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client"""
        from app.config import SUPABASE_URL, SUPABASE_ANON_KEY
        
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        
        from supabase import create_client
        self._client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        logger.info("Supabase client initialized")
    
    @property
    def client(self):
        """Get or initialize Supabase client"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def create_table_if_not_exists(self):
        """Create the vector store table with pgvector"""
        # Note: Run this SQL in Supabase dashboard
        sql = f"""
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Create table
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector({self.embedding_dim}),
            metadata JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create HNSW index for vector similarity search
        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
        ON {self.table_name} 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        
        -- Create indexes for filtering
        CREATE INDEX IF NOT EXISTS {self.table_name}_document_id_idx 
        ON {self.table_name}(document_id);
        
        -- Enable Row Level Security
        ALTER TABLE {self.table_name} ENABLE ROW LEVEL SECURITY;
        
        -- Create policy for anon reads
        CREATE POLICY "Allow anon reads" ON {self.table_name}
        FOR SELECT USING (true);
        
        -- Create policy for anon inserts
        CREATE POLICY "Allow anon inserts" ON {self.table_name}
        FOR INSERT WITH CHECK (true);
        
        -- Create policy for anon updates
        CREATE POLICY "Allow anon updates" ON {self.table_name}
        FOR UPDATE USING (true);
        
        -- Create policy for anon deletes
        CREATE POLICY "Allow anon deletes" ON {self.table_name}
        FOR DELETE USING (true);
        """
        
        logger.info("Run this SQL in Supabase SQL Editor to create vector table:\n%s", sql)
        return sql
    
    def add_documents(self, documents: List[Dict[str, Any]], 
                      document_id: str = None) -> List[str]:
        """Add documents with embeddings to the store"""
        if not documents:
            return []
        
        doc_id = document_id or str(uuid.uuid4())
        ids = []
        
        records = []
        for i, doc in enumerate(documents):
            record = {
                "document_id": doc_id,
                "chunk_index": i,
                "content": doc["content"],
                "embedding": doc["embedding"],
                "metadata": doc.get("metadata", {})
            }
            records.append(record)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.client.table(self.table_name).insert(batch).execute()
        
        logger.info(f"Added {len(records)} documents to vector store")
        return [doc_id] * len(records)
    
    def similarity_search(self, query_embedding: List[float], 
                          k: int = 5,
                          filter_metadata: Dict = None) -> List[Dict]:
        """Search for similar documents"""
        # Convert to proper format for Supabase
        query_vector = [float(x) for x in query_embedding]
        
        # Build query
        query = self.client.rpc(
            "match_documents",  # Need to create this function
            {
                "query_embedding": query_vector,
                "match_count": k,
                "filter_metadata": filter_metadata or {}
            }
        )
        
        # Fallback: simple similarity search
        try:
            response = query.execute()
            return response.data
        except Exception as e:
            logger.warning(f"RPC call failed, using fallback: {e}")
            return self._simple_similarity_search(query_vector, k, filter_metadata)
    
    def _simple_similarity_search(self, query_embedding: List[float],
                                   k: int = 5,
                                   filter_metadata: Dict = None) -> List[Dict]:
        """Fallback similarity search using raw SQL"""
        from app.config import SUPABASE_SERVICE_KEY
        
        if not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY required for fallback search")
        
        # Create service role client
        from supabase import create_client
        svc_client = create_client(
            os.getenv("SUPABASE_URL"),
            SUPABASE_SERVICE_KEY
        )
        
        # Use pgvector cosine similarity
        sql = f"""
        SELECT document_id, chunk_index, content, metadata,
               1 - (embedding <=> '{query_embedding}'::vector) as similarity
        FROM {self.table_name}
        {f"WHERE metadata @> '{filter_metadata}'::jsonb" if filter_metadata else ""}
        ORDER BY embedding <=> '{query_embedding}'::vector
        LIMIT {k}
        """
        
        response = svc_client.rpc("exec_sql", {"query": sql}).execute()
        
        # Parse results
        results = []
        for row in response.data:
            results.append({
                "document_id": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "metadata": row[3],
                "score": row[4]
            })
        
        return results
    
    def get_by_document_id(self, document_id: str) -> List[Dict]:
        """Get all chunks for a specific document"""
        response = self.client.table(self.table_name).select("*").eq(
            "document_id", document_id
        ).order("chunk_index").execute()
        
        return response.data
    
    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        response = self.client.table(self.table_name).delete().eq(
            "document_id", document_id
        ).execute()
        
        return len(response.data)
    
    def list_documents(self) -> List[str]:
        """List all unique document IDs"""
        response = self.client.table(self.table_name).select(
            "document_id", distinct=True
        ).execute()
        
        return [row["document_id"] for row in response.data]
    
    def get_document_count(self) -> int:
        """Get total chunk count"""
        response = self.client.table(self.table_name).select("*", count="exact").execute()
        return response.count or 0


# Global instance
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    """Get or create global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store