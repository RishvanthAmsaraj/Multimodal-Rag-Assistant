"""High-level retriever that ties the embedder and vector store together."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..processing.embedder import Embedder
from ..processing.chunker import Chunk
from .vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Embed → query → return ranked results."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "doc_type": c.doc_type,
                    **c.metadata,
                }
                for c in chunks
            ],
        )
        logger.info("Indexed %d chunks", len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        k = top_k or self.top_k
        query_vec = self.embedder.embed_query(query)
        results = self.vector_store.query(query_vec, top_k=k)
        if self.score_threshold > 0:
            results = [r for r in results if r.score >= self.score_threshold]
        return results

    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        results = self.retrieve(query, top_k=top_k)
        return [
            {
                "chunk_id": r.chunk_id,
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
            }
            for r in results
        ]