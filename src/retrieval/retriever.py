"""High-level retriever that ties the embedder and vector store together."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ..processing.embedder import Embedder
from ..processing.chunker import Chunk
from .vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)

# Common English words that add noise to keyword-overlap scoring.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "have", "has", "had", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "my", "me", "i", "you", "your", "they", "their",
    "it", "its", "and", "or", "not", "can", "could", "would", "should",
    "please", "about", "any", "tell", "show", "give", "list", "find",
    "get", "did", "am", "him", "her", "he", "she", "we", "us",
}


class Retriever:
    """Embed → query → return ranked results.

    When ``hybrid_weight`` > 0, dense cosine scores are blended with a
    lexical keyword-coverage score. Exact keyword matches matter a lot for
    personal/short documents (emails, names, section headers like
    "EXPERIENCE") where mean-pooled embeddings of long chunks lose the
    header signal entirely.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
        hybrid_weight: float = 0.0,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.hybrid_weight = max(0.0, min(1.0, hybrid_weight))

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
        # With hybrid scoring, over-fetch dense candidates so keyword matches
        # that rank outside the dense top-k can still surface. The extra
        # headroom also lets us drop duplicate chunks below without losing
        # coverage of the requested k.
        fetch_k = max(k, min(k * 4, 64)) if self.hybrid_weight > 0 else k * 2
        results = self.vector_store.query(query_vec, top_k=fetch_k)

        if self.hybrid_weight > 0:
            for r in results:
                r.score = (
                    (1 - self.hybrid_weight) * r.score
                    + self.hybrid_weight * self._lexical_score(query, r.text)
                )
            results.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate identical chunks (re-ingested documents double up):
        # keep the first occurrence of each unique text so one duplicate
        # set can't crowd real content out of the window.
        seen: set[str] = set()
        unique: List[RetrievalResult] = []
        for r in results:
            norm = " ".join(r.text.split()).strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            unique.append(r)
            if len(unique) >= k:
                break
        results = unique

        if self.score_threshold > 0:
            results = [r for r in results if r.score >= self.score_threshold]
        return results

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """Fraction of meaningful query terms present in the chunk.

        Simple, robust keyword coverage: for a resume query like
        "what work experience does Rishvanth have", a chunk headed
        "EXPERIENCE" scores 1/3 and a skills chunk scores 0 — exactly the
        signal dense embeddings lost inside long chunks.
        """
        q_terms = {
            t for t in re.findall(r"[a-z0-9]+", query.lower())
            if t not in _STOPWORDS and len(t) > 1
        }
        if not q_terms:
            return 0.0
        doc_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        return len(q_terms & doc_terms) / len(q_terms)

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