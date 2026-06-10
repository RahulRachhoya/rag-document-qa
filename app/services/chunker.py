"""Text Chunker - Semantic chunking with overlap"""
import re
from typing import List, Optional
import logging

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.services.loader import Document

logger = logging.getLogger(__name__)

class Chunk:
    """Represents a text chunk with metadata"""
    def __init__(self, content: str, metadata: dict, chunk_index: int = 0):
        self.page_content = content
        self.metadata = {**metadata, "chunk_index": chunk_index}

class TextChunker:
    """Split documents into semantic chunks"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents"""
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document using multiple strategies"""
        text = document.page_content
        metadata = document.metadata
        
        # Try semantic chunking first (by paragraphs/sections)
        if self._is_structured(text):
            chunks = self._semantic_chunk(text, metadata)
        else:
            # Fallback to fixed-size overlapping chunks
            chunks = self._fixed_chunk(text, metadata)
        
        # Add chunk indices
        return [
            Chunk(chunk.page_content, chunk.metadata, i) 
            for i, chunk in enumerate(chunks)
        ]
    
    def _is_structured(self, text: str) -> bool:
        """Check if text has clear structure (headings, paragraphs)"""
        # Check for common heading patterns
        heading_pattern = r'^(#{1,6}\s|.+\n={3,}|.+\n-{3,}|\d+\.\s+[A-Z])'
        return bool(re.search(heading_pattern, text, re.MULTILINE))
    
    def _semantic_chunk(self, text: str, metadata: dict) -> List[Chunk]:
        """Chunk by paragraphs and logical sections"""
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If single paragraph is too large, chunk it further
            if len(para) > self.chunk_size * 1.5:
                if current_chunk:
                    chunks.append(Chunk(current_chunk, metadata))
                    current_chunk = ""
                
                sub_chunks = self._fixed_chunk(para, metadata)
                chunks.extend(sub_chunks[:-1])  # Don't duplicate overlap
                current_chunk = sub_chunks[-1].page_content if sub_chunks else ""
            
            # Add paragraph to current chunk
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(Chunk(current_chunk, metadata))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                current_chunk = overlap_text + ("\n\n" if overlap_text else "") + para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(current_chunk, metadata))
        
        return chunks
    
    def _fixed_chunk(self, text: str, metadata: dict) -> List[Chunk]:
        """Fixed-size overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < text_length:
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(chunk_text, metadata))
            
            # Move with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_by_headings(self, text: str, metadata: dict) -> List[Chunk]:
        """Split document by heading boundaries (for TOC-like docs)"""
        chunks = []
        
        # Split by markdown headings
        heading_pattern = r'^#{1,6}\s+(.+)$'
        lines = text.split('\n')
        
        current_section = ""
        current_heading = "Introduction"
        chunk_index = 0
        
        for line in lines:
            if re.match(heading_pattern, line, re.MULTILINE):
                # Save previous section
                if current_section.strip():
                    chunks.append(Chunk(
                        current_section.strip(),
                        {**metadata, "heading": current_heading},
                        chunk_index
                    ))
                    chunk_index += 1
                
                current_heading = re.match(heading_pattern, line).group(1)
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        # Add final section
        if current_section.strip():
            chunks.append(Chunk(
                current_section.strip(),
                {**metadata, "heading": current_heading},
                chunk_index
            ))
        
        return chunks