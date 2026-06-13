"""Tests for the DocumentLoader (multi-format support)."""

from __future__ import annotations

import pytest

from rag_qa.services.loader import DocumentLoader


class TestDocumentLoader:
    """Feature: Load text from supported file types."""

    def test_load_txt(self, sample_txt_path: str):
        loader = DocumentLoader()
        text = loader.load(sample_txt_path)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "retrieval augmented generation" in text.lower()

    # Add real PDF/DOCX tests when you place minimal valid files in tests/data/
    # def test_load_pdf(self):
    #     loader = DocumentLoader()
    #     text = loader.load("tests/data/sample.pdf")
    #     assert "expected text from pdf" in text.lower()

    def test_unsupported_raises(self, tmp_path):
        bad = tmp_path / "bad.xyz"
        bad.write_text("hello")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(str(bad))
