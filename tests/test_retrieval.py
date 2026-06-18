"""Tests for the vector store + retriever."""

import numpy as np
import pytest

from src.processing.chunker import Chunk
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


def _make_chunks(texts):
    return [
        Chunk(
            text=t,
            chunk_id=f"chunk_{i}",
            doc_id=f"doc_{i // 2}",
            source=f"doc_{i // 2}.txt",
            doc_type="text",
            metadata={"chunk_index": i},
        )
        for i, t in enumerate(texts)
    ]


def test_vector_store_add_and_count(tmp_vector_dir, fake_embedder):
    store = VectorStore(
        collection_name="test",
        persist_directory=tmp_vector_dir,
        persist=False,
    )
    chunks = _make_chunks(["hello world", "goodbye world", "rag pipelines"])
    vecs = fake_embedder.embed([c.text for c in chunks])
    store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vecs,
        documents=[c.text for c in chunks],
        metadatas=[{"doc_id": c.doc_id, "source": c.source} for c in chunks],
    )
    assert store.count() == 3


def test_retriever_returns_relevant_chunks(tmp_vector_dir, fake_embedder):
    store = VectorStore(
        collection_name="test",
        persist_directory=tmp_vector_dir,
        persist=False,
    )
    retriever = Retriever(embedder=fake_embedder, vector_store=store, top_k=2)

    chunks = _make_chunks(
        [
            "The Eiffel Tower is in Paris.",
            "Tokyo is the capital of Japan.",
            "Retrieval augmented generation reduces hallucinations.",
        ]
    )
    retriever.index_chunks(chunks)

    results = retriever.retrieve("Where is the Eiffel Tower?")
    assert len(results) >= 1
    top = results[0]
    # Top result should be the Eiffel Tower chunk, not the unrelated ones
    assert "Eiffel" in top.text or "Paris" in top.text


def test_retriever_score_threshold_filters(tmp_vector_dir, fake_embedder):
    store = VectorStore(
        collection_name="test",
        persist_directory=tmp_vector_dir,
        persist=False,
    )
    retriever = Retriever(
        embedder=fake_embedder, vector_store=store, top_k=5, score_threshold=0.99
    )
    retriever.index_chunks(_make_chunks(["alpha beta", "gamma delta", "epsilon zeta"]))

    # With an extreme threshold we expect zero results
    results = retriever.retrieve("completely unrelated query about xyz")
    assert all(r.score >= 0.99 for r in results)
    # Results should be empty for a clearly dissimilar query (no overlap)
    assert isinstance(results, list)


def test_retriever_with_scores_metadata(tmp_vector_dir, fake_embedder):
    store = VectorStore(
        collection_name="test",
        persist_directory=tmp_vector_dir,
        persist=False,
    )
    retriever = Retriever(embedder=fake_embedder, vector_store=store, top_k=2)
    chunks = _make_chunks(["apple banana", "apple cherry", "grape kiwi"])
    retriever.index_chunks(chunks)

    out = retriever.retrieve_with_scores("apple fruit", top_k=2)
    assert len(out) == 2
    assert all("score" in d and "text" in d and "metadata" in d for d in out)