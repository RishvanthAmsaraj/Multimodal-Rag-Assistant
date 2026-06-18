"""Plain text / markdown ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

DEFAULT_EXTENSIONS = [".txt", ".md", ".markdown", ".rst", ".log"]


class TextLoader:
    """Load plain-text or markdown files."""

    def __init__(self, encoding: str = "utf-8", extensions: List[str] | None = None):
        self.encoding = encoding
        self.extensions = [e.lower() for e in (extensions or DEFAULT_EXTENSIONS)]

    def load(self, file_path: PathLike) -> Dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        if file_path.suffix.lower() not in self.extensions:
            raise ValueError(
                f"Unsupported extension {file_path.suffix}. Allowed: {self.extensions}"
            )

        text = file_path.read_text(encoding=self.encoding, errors="replace")
        logger.info("Loaded text file %s: %d chars", file_path.name, len(text))

        return {
            "source": str(file_path),
            "filename": file_path.name,
            "type": "text",
            "text": text,
            "full_text": text,
            "metadata": {
                "size_bytes": file_path.stat().st_size,
                "encoding": self.encoding,
            },
        }

    def load_batch(self, file_paths: List[PathLike]) -> List[Dict]:
        return [self.load(p) for p in file_paths]