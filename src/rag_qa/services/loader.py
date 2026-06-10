"""Document loader: PDF, DOCX, and plain-text support."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load text content from PDF, DOCX, or TXT files."""

    SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

    def load(self, file_path: str) -> str:
        """Return the full text of the document at *file_path*."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {self.SUPPORTED}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".docx":
            return self._load_docx(path)
        return self._load_text(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF loading: pip install pypdf") from exc

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        full_text = "\n".join(pages)
        logger.debug("PDF loaded: %s (%d pages, %d chars)", path.name, len(reader.pages), len(full_text))
        return full_text

    def _load_docx(self, path: Path) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "python-docx is required for DOCX loading: pip install python-docx"
            ) from exc

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        logger.debug("DOCX loaded: %s (%d paragraphs)", path.name, len(paragraphs))
        return full_text

    def _load_text(self, path: Path) -> str:
        full_text = path.read_text(encoding="utf-8", errors="replace")
        logger.debug("Text loaded: %s (%d chars)", path.name, len(full_text))
        return full_text
