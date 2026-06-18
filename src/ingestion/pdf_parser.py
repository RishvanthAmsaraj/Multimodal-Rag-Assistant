"""PDF document ingestion.

Extracts text (and optionally metadata) from PDF files using
PyPDF2 / pypdf. Designed to be fault-tolerant: if a page fails
to parse we log and continue instead of crashing the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class PDFParser:
    """Robust PDF text extractor."""

    def __init__(self, max_pages: int = 0, library: str = "pypdf"):
        self.max_pages = max_pages
        self.library = library
        logger.info("PDFParser initialised (library=%s, max_pages=%s)", library, max_pages)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self, file_path: PathLike) -> Dict:
        """Parse a PDF file and return a structured dict.

        Returns
        -------
        dict
            {
              "source": "...",
              "type": "pdf",
              "pages": [{"page": 1, "text": "...", "metadata": {...}}, ...],
              "full_text": "..."
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if self.library == "pypdf":
            pages = self._parse_with_pypdf(file_path)
        else:
            # pymupdf fallback path (kept light to avoid hard dep)
            pages = self._parse_with_pymupdf(file_path)

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        logger.info("Parsed PDF %s: %d pages, %d chars", file_path.name, len(pages), len(full_text))

        return {
            "source": str(file_path),
            "filename": file_path.name,
            "type": "pdf",
            "pages": pages,
            "full_text": full_text,
            "metadata": {
                "num_pages": len(pages),
                "size_bytes": file_path.stat().st_size,
            },
        }

    def parse_batch(self, file_paths: List[PathLike]) -> List[Dict]:
        """Parse multiple PDFs."""
        return [self.parse(p) for p in file_paths]

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _parse_with_pypdf(self, file_path: Path) -> List[Dict]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF parsing. Install with `pip install pypdf`."
            ) from exc

        reader = PdfReader(str(file_path))
        n_pages = len(reader.pages) if self.max_pages == 0 else min(self.max_pages, len(reader.pages))

        pages: List[Dict] = []
        for i in range(n_pages):
            try:
                text = reader.pages[i].extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to extract page %d from %s: %s", i + 1, file_path.name, exc)
                text = ""
            pages.append(
                {
                    "page": i + 1,
                    "text": text.strip(),
                    "metadata": self._safe_doc_info(reader),
                }
            )
        return pages

    def _parse_with_pymupdf(self, file_path: Path) -> List[Dict]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError("PyMuPDF not installed. Set library='pypdf' in config.") from exc

        doc = fitz.open(str(file_path))
        n_pages = len(doc) if self.max_pages == 0 else min(self.max_pages, len(doc))
        pages: List[Dict] = []
        for i in range(n_pages):
            text = doc[i].get_text("text") or ""
            pages.append({"page": i + 1, "text": text.strip(), "metadata": {}})
        doc.close()
        return pages

    @staticmethod
    def _safe_doc_info(reader) -> Dict:
        try:
            info = reader.metadata or {}
            return {k: str(v) for k, v in info.items() if v is not None}
        except Exception:  # noqa: BLE001
            return {}