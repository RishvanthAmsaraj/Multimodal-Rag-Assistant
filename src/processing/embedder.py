"""Embedding generation using sentence-transformers.

The embedder wraps a HuggingFace sentence-transformers model
(default ``all-MiniLM-L6-v2``) and exposes a simple batch API.
It also provides a ``numpy``-backed in-memory cache so repeated
queries don't hit the model twice.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Generate dense vector embeddings for text chunks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self.use_cache = use_cache
        self._cache: dict[str, np.ndarray] = {}

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. Run `pip install sentence-transformers`."
            ) from exc

        logger.info("Loading embedding model %s on %s ...", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedder ready (dim=%d)", self.dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of strings. Returns ``(n, dim)`` float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors = [None] * len(texts)
        missing_idx: List[int] = []
        missing_texts: List[str] = []

        for i, t in enumerate(texts):
            key = self._cache_key(t)
            if self.use_cache and key in self._cache:
                vectors[i] = self._cache[key]
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            new_vecs = self.model.encode(
                missing_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ).astype(np.float32)

            for j, idx in enumerate(missing_idx):
                vectors[idx] = new_vecs[j]
                if self.use_cache:
                    self._cache[self._cache_key(missing_texts[j])] = new_vecs[j]

        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns ``(dim,)`` vector."""
        return self.embed([query])[0]

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()