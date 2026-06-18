"""Tests for the text chunker."""

from src.processing.chunker import TextChunker


def test_recursive_chunk_basic():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10, strategy="recursive")
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    chunks = chunker._chunk_text(
        text,
        doc_id="doc1",
        source="test.txt",
        doc_type="text",
        extra_metadata={},
    )
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)
    # IDs are unique + traceable
    assert all(c.doc_id == "doc1" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_fixed_chunk_overlap():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5, strategy="fixed")
    text = "a" * 100
    chunks = chunker._chunk_text(text, doc_id="d", source="t", doc_type="text", extra_metadata={})
    assert len(chunks) >= 4  # 100/15 ≈ 7 with overlap


def test_sentence_chunk_respects_size():
    chunker = TextChunker(chunk_size=40, chunk_overlap=0, strategy="sentence")
    text = "A short one. " + "Another sentence here. " * 10
    chunks = chunker._chunk_text(text, doc_id="d", source="t", doc_type="text", extra_metadata={})
    assert all(len(c.text) <= 80 for c in chunks)  # allow some slack


def test_chunk_document_pdf_preserves_page_metadata():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20, strategy="recursive")
    doc = {
        "source": "x.pdf",
        "type": "pdf",
        "pages": [
            {"page": 1, "text": "Page one content " * 30, "metadata": {}},
            {"page": 2, "text": "Page two content " * 30, "metadata": {}},
        ],
    }
    chunks = chunker.chunk_document(doc)
    pages_seen = {c.metadata.get("page") for c in chunks}
    assert pages_seen == {1, 2}


def test_invalid_overlap_raises():
    import pytest
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_chunk_documents_returns_chunks_with_ids():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10, strategy="recursive")
    docs = [
        {"source": "a.txt", "type": "text", "text": "Hello world. " * 20, "full_text": "Hello world. " * 20},
        {"source": "b.txt", "type": "text", "text": "Goodbye world. " * 20, "full_text": "Goodbye world. " * 20},
    ]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) > 0
    assert any(c.source == "a.txt" for c in chunks)
    assert any(c.source == "b.txt" for c in chunks)