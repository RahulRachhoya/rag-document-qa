"""Text chunker using recursive character splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Separators tried in order (most to least structural)
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


@dataclass
class Chunk:
    """A text chunk with positional metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """Split text with recursive separator fallback, 512/64 defaults."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or _DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split *text* into overlapping chunks."""
        if not text.strip():
            return []

        raw_chunks = self._recursive_split(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        chunks: list[Chunk] = []
        cursor = 0
        for idx, chunk_text in enumerate(merged):
            start = text.find(chunk_text, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk_text)
            cursor = max(cursor, start)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    metadata=dict(metadata or {}),
                )
            )

        logger.debug("Split into %d chunks (size=%d, overlap=%d)", len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split using the first separator that works."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:]

        if sep:
            splits = text.split(sep)
        else:
            # Character-level split as last resort
            splits = list(text)

        result: list[str] = []
        current = ""
        for part in splits:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # part itself may be too large - recurse
                if len(part) > self.chunk_size and remaining_seps:
                    result.extend(self._recursive_split(part, remaining_seps))
                elif part.strip():
                    result.append(part)
                current = ""

        if current:
            result.append(current)

        return [r for r in result if r.strip()]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Merge small pieces and add overlap between adjacent chunks."""
        if not pieces:
            return []

        merged: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len > self.chunk_size and current_parts:
                chunk_text = " ".join(current_parts)
                merged.append(chunk_text)
                # Keep overlap: trim from the front until within overlap budget
                while current_parts and current_len > self.chunk_overlap:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + 1
            current_parts.append(piece)
            current_len += piece_len + (1 if current_parts else 0)

        if current_parts:
            merged.append(" ".join(current_parts))

        return merged
