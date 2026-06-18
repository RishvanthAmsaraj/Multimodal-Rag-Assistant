"""Tests for the evaluation metrics module."""

import math

import pytest

from src.evaluation.metrics import EvalSample, RAGEvaluator


def test_precision_recall_basic():
    s = EvalSample(
        question="q",
        relevant_chunk_ids=["a", "b"],
        retrieved_chunk_ids=["a", "c", "d"],
    )
    ev = RAGEvaluator()
    assert ev.retrieval_precision_at_k(s, k=3) == pytest.approx(1 / 3)
    assert ev.retrieval_recall_at_k(s, k=3) == pytest.approx(1 / 2)


def test_precision_k_truncates():
    s = EvalSample(
        question="q",
        relevant_chunk_ids=["a"],
        retrieved_chunk_ids=["a", "b", "c"],
    )
    ev = RAGEvaluator()
    assert ev.retrieval_precision_at_k(s, k=1) == 1.0
    assert ev.retrieval_precision_at_k(s, k=3) == pytest.approx(1 / 3)


def test_mrr_first_relevant_position():
    s1 = EvalSample(question="q", relevant_chunk_ids=["b"], retrieved_chunk_ids=["a", "b", "c"])
    s2 = EvalSample(question="q", relevant_chunk_ids=["a"], retrieved_chunk_ids=["a", "b", "c"])
    s3 = EvalSample(question="q", relevant_chunk_ids=["z"], retrieved_chunk_ids=["a", "b", "c"])

    ev = RAGEvaluator()
    assert ev.mrr(s1) == pytest.approx(1 / 2)
    assert ev.mrr(s2) == pytest.approx(1.0)
    assert ev.mrr(s3) == 0.0


def test_answer_faithfulness_supported_sentences():
    s = EvalSample(
        question="q",
        answer="Paris is in France. Mars is a planet made of chocolate.",
        retrieved_texts=["Paris is the capital of France."],
    )
    ev = RAGEvaluator()
    # First sentence is supported, second isn't => 0.5
    assert ev.answer_faithfulness(s) == pytest.approx(0.5)


def test_evaluate_aggregates():
    samples = [
        EvalSample(
            question="What is RAG?",
            relevant_chunk_ids=["c1"],
            retrieved_chunk_ids=["c1", "c2"],
            answer="RAG is retrieval augmented generation.",
            retrieved_texts=["Retrieval augmented generation uses a vector store."],
        ),
        EvalSample(
            question="Where is Paris?",
            relevant_chunk_ids=["c3"],
            retrieved_chunk_ids=["c4", "c3"],
            answer="Paris is in France.",
            retrieved_texts=["Paris is the capital of France."],
        ),
    ]
    report = RAGEvaluator().evaluate(samples, k=2)
    assert "precision@2" in report.metrics
    assert "recall@2" in report.metrics
    assert "mrr" in report.metrics
    assert "answer_faithfulness" in report.metrics
    assert all(0.0 <= v <= 1.0 for v in report.metrics.values())


def test_empty_sample_returns_zero():
    s = EvalSample(question="q")
    ev = RAGEvaluator()
    assert ev.retrieval_precision_at_k(s) == 0.0
    assert ev.retrieval_recall_at_k(s) == 0.0
    assert ev.answer_faithfulness(s) == 0.0
    assert ev.context_precision(s) == 0.0