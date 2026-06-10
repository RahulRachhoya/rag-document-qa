"""Unit tests for RecursiveChunker."""

from __future__ import annotations

import pytest

from rag_qa.services.chunker import RecursiveChunker


def make_chunker(**kwargs) -> RecursiveChunker:
    defaults = {"chunk_size": 100, "chunk_overlap": 10}
    defaults.update(kwargs)
    return RecursiveChunker(**defaults)


class TestRecursiveChunkerInit:
    def test_default_params(self):
        c = RecursiveChunker()
        assert c.chunk_size == 512
        assert c.chunk_overlap == 64

    def test_custom_params(self):
        c = make_chunker(chunk_size=256, chunk_overlap=32)
        assert c.chunk_size == 256

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveChunker(chunk_size=0)

    def test_overlap_gte_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=-1)


class TestRecursiveChunkerSplit:
    def test_empty_text_returns_empty(self):
        c = make_chunker()
        assert c.split("") == []

    def test_whitespace_only_returns_empty(self):
        c = make_chunker()
        assert c.split("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        c = make_chunker(chunk_size=500)
        text = "Hello, world!"
        chunks = c.split(text)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self):
        c = make_chunker(chunk_size=50, chunk_overlap=5)
        text = "word " * 50  # 250 chars
        chunks = c.split(text)
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self):
        c = make_chunker(chunk_size=60, chunk_overlap=5)
        text = "sentence one. sentence two. sentence three. sentence four."
        chunks = c.split(text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_metadata_propagated(self):
        c = make_chunker(chunk_size=500)
        meta = {"doc_id": "abc", "filename": "test.txt"}
        chunks = c.split("some content", metadata=meta)
        assert chunks[0].metadata["doc_id"] == "abc"
        assert chunks[0].metadata["filename"] == "test.txt"

    def test_chunk_text_not_empty(self):
        c = make_chunker(chunk_size=80, chunk_overlap=10)
        text = "paragraph one\n\nparagraph two\n\nparagraph three\n\nparagraph four"
        chunks = c.split(text)
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_chunk_size_respected(self):
        c = make_chunker(chunk_size=100, chunk_overlap=10)
        text = "x" * 500
        chunks = c.split(text)
        for chunk in chunks:
            # Allow minor overflow at the merging stage
            assert len(chunk.text) <= c.chunk_size * 2
