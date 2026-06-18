"""Evaluation metrics for the RAG pipeline.

Implemented metrics:

* ``retrieval_precision_at_k``  - fraction of top-k retrieved chunks that are relevant.
* ``retrieval_recall_at_k``     - fraction of relevant chunks that were retrieved in top-k.
* ``answer_relevance``          - cosine similarity between question embedding and answer embedding.
* ``answer_faithfulness``       - fraction of answer sentences supported by retrieved context.
* ``context_precision``         - precision of the retrieved context set as a whole.

These are intentionally lightweight (no LLM-as-judge dependency) so
the eval suite runs offline in CI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """One (question, expected_chunks, expected_answer) eval sample."""

    question: str
    relevant_chunk_ids: List[str] = field(default_factory=list)
    reference_answer: str = ""
    answer: str = ""
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_texts: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated metrics for a batch of samples."""

    metrics: Dict[str, float]
    per_sample: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Evaluation Report", "=" * 40]
        for k, v in self.metrics.items():
            lines.append(f"  {k:30s} {v:.4f}")
        return "\n".join(lines)


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


class RAGEvaluator:
    """Compute retrieval + answer quality metrics."""

    def __init__(self, embedder=None):
        # Optional: embedder for answer_relevance (cosine sim w/ question).
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------
    @staticmethod
    def retrieval_precision_at_k(sample: EvalSample, k: int = 5) -> float:
        if not sample.retrieved_chunk_ids:
            return 0.0
        top_k = sample.retrieved_chunk_ids[:k]
        relevant = set(sample.relevant_chunk_ids)
        if not relevant:
            return 0.0
        hits = sum(1 for cid in top_k if cid in relevant)
        return hits / max(1, len(top_k))

    @staticmethod
    def retrieval_recall_at_k(sample: EvalSample, k: int = 5) -> float:
        if not sample.relevant_chunk_ids:
            return 0.0
        top_k = sample.retrieved_chunk_ids[:k]
        relevant = set(sample.relevant_chunk_ids)
        hits = sum(1 for cid in top_k if cid in relevant)
        return hits / len(relevant)

    @staticmethod
    def mrr(sample: EvalSample) -> float:
        """Mean Reciprocal Rank — 1/rank of the first relevant doc."""
        relevant = set(sample.relevant_chunk_ids)
        for i, cid in enumerate(sample.retrieved_chunk_ids, start=1):
            if cid in relevant:
                return 1.0 / i
        return 0.0

    # ------------------------------------------------------------------
    # Answer metrics
    # ------------------------------------------------------------------
    def answer_relevance(self, sample: EvalSample) -> float:
        """Cosine similarity between question and answer embeddings.

        Falls back to token-overlap if no embedder is supplied.
        """
        if self.embedder is not None:
            vecs = self.embedder.embed([sample.question, sample.answer])
            q, a = vecs[0], vecs[1]
            denom = (np.linalg.norm(q) * np.linalg.norm(a))
            if denom == 0:
                return 0.0
            return float(np.dot(q, a) / denom)
        # Fallback: token Jaccard
        return self._jaccard(sample.question, sample.answer)

    def answer_faithfulness(self, sample: EvalSample) -> float:
        """Fraction of answer sentences whose tokens are mostly in the
        retrieved context. Heuristic; no LLM judge required."""
        if not sample.answer or not sample.retrieved_texts:
            return 0.0
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", sample.answer) if s.strip()]
        if not sentences:
            return 0.0

        ctx_tokens = set()
        for t in sample.retrieved_texts:
            ctx_tokens |= _tokenize(t)

        supported = 0
        for sent in sentences:
            sent_tokens = _tokenize(sent)
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & ctx_tokens)
            if overlap / len(sent_tokens) >= 0.5:
                supported += 1
        return supported / len(sentences)

    @staticmethod
    def context_precision(sample: EvalSample) -> float:
        """Precision over the full retrieved set (not just top-k)."""
        if not sample.retrieved_chunk_ids or not sample.relevant_chunk_ids:
            return 0.0
        relevant = set(sample.relevant_chunk_ids)
        hits = sum(1 for cid in sample.retrieved_chunk_ids if cid in relevant)
        return hits / len(sample.retrieved_chunk_ids)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def evaluate(self, samples: List[EvalSample], k: int = 5) -> EvalReport:
        per_sample: List[Dict] = []
        agg: Dict[str, List[float]] = {
            f"precision@{k}": [],
            f"recall@{k}": [],
            "mrr": [],
            "answer_relevance": [],
            "answer_faithfulness": [],
            "context_precision": [],
        }

        for s in samples:
            row = {
                "question": s.question,
                f"precision@{k}": self.retrieval_precision_at_k(s, k=k),
                f"recall@{k}": self.retrieval_recall_at_k(s, k=k),
                "mrr": self.mrr(s),
                "answer_relevance": self.answer_relevance(s),
                "answer_faithfulness": self.answer_faithfulness(s),
                "context_precision": self.context_precision(s),
            }
            per_sample.append(row)
            for m, v in row.items():
                if m in agg:
                    agg[m].append(v)

        means = {m: float(np.mean(vs)) if vs else 0.0 for m, vs in agg.items()}
        return EvalReport(metrics=means, per_sample=per_sample)

    # ------------------------------------------------------------------
    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = _tokenize(a), _tokenize(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


def run_evaluation(
    pipeline,
    samples: Sequence[Dict],
    k: int = 5,
    embedder=None,
) -> EvalReport:
    """Convenience runner: feed raw {question, relevant_chunk_ids, reference_answer}
    dicts through the pipeline and evaluate.
    """
    evaluator = RAGEvaluator(embedder=embedder or pipeline.embedder)
    eval_samples: List[EvalSample] = []

    for s in samples:
        result = pipeline.query(s["question"])
        eval_samples.append(
            EvalSample(
                question=s["question"],
                relevant_chunk_ids=s.get("relevant_chunk_ids", []),
                reference_answer=s.get("reference_answer", ""),
                answer=result["answer"],
                retrieved_chunk_ids=[src["chunk_id"] for src in result.get("sources", [])],
                retrieved_texts=[src["text"] for src in result.get("sources", [])],
            )
        )

    return evaluator.evaluate(eval_samples, k=k)