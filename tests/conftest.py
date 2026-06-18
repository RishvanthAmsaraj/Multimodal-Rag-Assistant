"""Shared pytest fixtures for the RAG test suite.

Most fixtures avoid loading real heavy models — they use small
fake components (e.g. hash-based vectors) so the suite is fast
and runs offline in CI.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Ensure the project root is on sys.path so `src.*` imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Embedding fakes
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Vectors are derived from a hashed bag-of-words representation
    so semantically similar texts yield similar vectors.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.model_name = "fake-embedder"

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in set(t.lower().split()):
                h = abs(hash(tok)) % self.dim
                out[i, h] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    def embed_query(self, q: str) -> np.ndarray:
        return self.embed([q])[0]

    @property
    def cache_size(self) -> int:
        return 0

    def clear_cache(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_embedder():
    return FakeEmbedder(dim=64)


@pytest.fixture
def tmp_vector_dir(tmp_path):
    """Provide a temporary, throwaway vector-store directory."""
    d = tmp_path / "chroma_test"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_pdf_text():
    """Pretend we extracted this text from a PDF page."""
    return (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed "
        "and built the tower. Locally nicknamed La dame de fer, it was "
        "constructed from 1887 to 1889 as the centerpiece of the 1889 "
        "World's Fair. Although initially criticised by some of France's "
        "leading artists and intellectuals for its design, it has since "
        "become a global cultural icon of France."
    )


@pytest.fixture
def sample_documents(sample_pdf_text):
    """A small mixed-type corpus."""
    return [
        {
            "source": "tests/data/eiffel.txt",
            "type": "text",
            "text": sample_pdf_text,
            "full_text": sample_pdf_text,
            "metadata": {"author": "Wikipedia"},
        },
        {
            "source": "tests/data/ai.txt",
            "type": "text",
            "text": (
                "Retrieval-Augmented Generation (RAG) is a technique that "
                "augments an LLM's generation with external knowledge "
                "retrieved from a vector store. It reduces hallucinations "
                "and allows the model to answer questions about private "
                "or recent documents without retraining."
            ),
            "full_text": (
                "Retrieval-Augmented Generation (RAG) is a technique that "
                "augments an LLM's generation with external knowledge "
                "retrieved from a vector store. It reduces hallucinations "
                "and allows the model to answer questions about private "
                "or recent documents without retraining."
            ),
            "metadata": {"author": "Test"},
        },
    ]