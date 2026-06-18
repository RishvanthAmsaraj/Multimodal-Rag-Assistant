"""Text chunking strategies.

We support three strategies out of the box:

* ``recursive``  - LangChain-style recursive splitter (default, robust)
* ``fixed``      - fixed character windows with overlap
* ``sentence``   - one chunk per sentence(s), respecting max size

All strategies preserve source metadata so chunks can be traced
back to their originating document + page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text plus provenance metadata."""

    text: str
    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    metadata: Dict = field(default_factory=dict)


class TextChunker:
    """Split documents into overlapping chunks suitable for embedding."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "recursive",
        separators: List[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        logger.info(
            "TextChunker ready (strategy=%s, size=%d, overlap=%d)",
            strategy, chunk_size, chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk_document(self, document: Dict, doc_id: str | None = None) -> List[Chunk]:
        """Chunk a single document dict returned by an ingestion module.

        For PDFs we chunk per page (so page metadata is preserved).
        For text/image docs we chunk the full text.
        """
        doc_type = document.get("type", "text")
        source = document.get("source") or document.get("filename", "<unknown>")
        doc_id = doc_id or f"{doc_type}:{source}"

        if doc_type == "pdf" and document.get("pages"):
            chunks: List[Chunk] = []
            for page in document["pages"]:
                page_text = page.get("text", "")
                if not page_text.strip():
                    continue
                chunks.extend(
                    self._chunk_text(
                        page_text,
                        doc_id=doc_id,
                        source=source,
                        doc_type=doc_type,
                        extra_metadata={"page": page.get("page"), **page.get("metadata", {})},
                    )
                )
            return chunks

        full_text = document.get("full_text") or document.get("text", "")
        return self._chunk_text(
            full_text,
            doc_id=doc_id,
            source=source,
            doc_type=doc_type,
            extra_metadata=document.get("metadata", {}),
        )

    def chunk_documents(self, documents: List[Dict]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for i, doc in enumerate(documents):
            doc_id = f"{doc.get('type', 'text')}:{doc.get('source', f'doc_{i}')}"
            all_chunks.extend(self.chunk_document(doc, doc_id=doc_id))
        logger.info("Chunked %d documents into %d chunks", len(documents), len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------
    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        source: str,
        doc_type: str,
        extra_metadata: Dict,
    ) -> List[Chunk]:
        if self.strategy == "fixed":
            raw_pieces = self._fixed_chunk(text)
        elif self.strategy == "sentence":
            raw_pieces = self._sentence_chunk(text)
        else:
            raw_pieces = self._recursive_chunk(text)

        chunks: List[Chunk] = []
        for idx, piece in enumerate(raw_pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{doc_id}::chunk_{idx:04d}"
            chunks.append(
                Chunk(
                    text=piece,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source=source,
                    doc_type=doc_type,
                    metadata={"chunk_index": idx, **extra_metadata},
                )
            )
        return chunks

    def _fixed_chunk(self, text: str) -> List[str]:
        step = self.chunk_size - self.chunk_overlap
        return [text[i : i + self.chunk_size] for i in range(0, max(1, len(text)), step)]

    def _sentence_chunk(self, text: str) -> List[str]:
        # Naive sentence split; OK for English without NLTK dep.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 <= self.chunk_size:
                buf = (buf + " " + sent).strip()
            else:
                if buf:
                    chunks.append(buf)
                # If a single sentence exceeds chunk_size, hard-split it.
                if len(sent) > self.chunk_size:
                    for i in range(0, len(sent), self.chunk_size):
                        chunks.append(sent[i : i + self.chunk_size])
                    buf = ""
                else:
                    buf = sent
        if buf:
            chunks.append(buf)
        return chunks

    def _recursive_chunk(self, text: str) -> List[str]:
        """Recursive splitter — tries separators in order, falling back to
        character-level slicing for the un-splittable tail."""
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Final fallback: hard slice with overlap.
            step = self.chunk_size - self.chunk_overlap
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        sep, rest = separators[0], separators[1:]
        pieces = text.split(sep) if sep else list(text)
        pieces = [p + sep for p in pieces[:-1]] + ([pieces[-1]] if pieces else [])

        chunks: List[str] = []
        buf = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(self._recursive_split(piece, rest))
            elif len(buf) + len(piece) <= self.chunk_size:
                buf += piece
            else:
                chunks.append(buf)
                buf = piece

        if buf:
            chunks.append(buf)

        # Apply overlap by sliding a tail from the previous chunk into the next.
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped: List[str] = []
            for i, c in enumerate(chunks):
                if i == 0:
                    overlapped.append(c)
                    continue
                tail = overlapped[-1][-self.chunk_overlap :]
                merged = (tail + c)
                if len(merged) <= self.chunk_size * 1.5:  # allow mild overshoot
                    overlapped[-1] = merged
                else:
                    overlapped.append(c)
            chunks = overlapped

        return chunks