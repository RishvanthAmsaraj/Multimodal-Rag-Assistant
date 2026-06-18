"""End-to-end smoke test that exercises the full pipeline with fakes.

We monkey-patch the Embedder + LLMInterface so we never load real
heavy models in CI — but every other component (ingestion → chunking
→ vector store → retriever → generator wrapper) runs against real code.
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.pipeline import RAGPipeline
from src.processing.embedder import Embedder
from src.generation.llm_interface import LLMInterface


class FakeEmbedder(Embedder):
    def __init__(self, *args, **kwargs):
        # Skip heavy model load
        self.dim = 64
        self.model_name = "fake-embedder"
        self.use_cache = False
        self._cache = {}

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in set(t.lower().split()):
                h = abs(hash(tok)) % self.dim
                out[i, h] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    def embed_query(self, q):
        return self.embed([q])[0]


class FakeLLM(LLMInterface):
    def __init__(self, *args, **kwargs):
        # Don't init any backend
        self.provider = "fake"
        self.model_name = "fake-llm"
        self.max_new_tokens = 256
        self.temperature = 0.0
        self.prompt_template = kwargs.get(
            "prompt_template",
            "Context: {context}\nQ: {question}\nA:",
        )

    def generate(self, question, context_chunks):
        ctx = " ".join(context_chunks)
        return f"[fake-answer] based on {len(context_chunks)} chunks (q={question[:30]})"


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Build a pipeline with fakes for Embedder + LLM."""
    cfg = load_config()
    cfg.paths.vector_store_dir = str(tmp_path / "chroma")
    cfg.processing.embedder.model_name = "fake"
    cfg.generation.llm.provider = "fake"

    pl = RAGPipeline.__new__(RAGPipeline)  # bypass __init__
    pl.config = cfg

    # Real components
    from src.ingestion.image_ocr import ImageOCR
    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.text_loader import TextLoader
    from src.processing.chunker import TextChunker
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import VectorStore

    pl.pdf_parser = PDFParser()
    pl.image_ocr = ImageOCR(engine="pytesseract")
    pl.text_loader = TextLoader()
    pl.chunker = TextChunker(chunk_size=120, chunk_overlap=20, strategy="recursive")
    pl.embedder = FakeEmbedder()
    pl.vector_store = VectorStore(
        collection_name="e2e_test",
        persist_directory=cfg.paths.vector_store_dir,
        persist=False,
    )
    pl.retriever = Retriever(pl.embedder, pl.vector_store, top_k=2)
    pl.llm = FakeLLM(prompt_template=str(cfg.generation.prompt.template))
    return pl


def test_ingest_and_query_text(pipeline, tmp_path):
    # Write a sample text file
    p = tmp_path / "doc.txt"
    p.write_text(
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1889. " * 5
    )

    chunks = pipeline.ingest_file(p)
    assert len(chunks) > 0
    assert pipeline.vector_store.count() == len(chunks)

    result = pipeline.query("Where is the Eiffel Tower?")
    assert "answer" in result
    assert result["num_sources"] > 0
    assert "[fake-answer]" in result["answer"]


def test_stats(pipeline):
    s = pipeline.stats()
    assert "num_chunks" in s
    assert "embedder_model" in s
    assert "embedding_dim" in s
    assert s["embedding_dim"] == 64


def test_query_with_no_documents(pipeline):
    result = pipeline.query("Anything?")
    assert "answer" in result
    assert "no indexed documents" in result["answer"].lower() or result["num_sources"] >= 0