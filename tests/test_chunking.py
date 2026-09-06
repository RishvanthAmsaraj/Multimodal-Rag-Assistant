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


def test_recursive_chunk_preserves_all_content():
    """Regression: a long document must not lose any content to chunking.

    The previous overlap implementation merged chunks into their neighbour
    and replaced the accumulator, collapsing a multi-chunk document into a
    single chunk holding only its tail. Every section must survive.
    """
    chunker = TextChunker(chunk_size=120, chunk_overlap=20, strategy="recursive")
    sections = [
        "Alpha beta gamma delta epsilon. " * 20,
        "Zeta eta theta iota kappa lambda mu. " * 20,
    ]
    text = "".join(sections)
    chunks = chunker._chunk_text(
        text, doc_id="doc2", source="t.txt", doc_type="text", extra_metadata={}
    )
    assert len(chunks) > 1, "long text should produce multiple chunks"
    # Overlap bridges duplicate the seams, so check that both ends of every
    # section survive somewhere in the output. The collapse bug kept only
    # the document's final tail, so the first section's head check alone
    # would have caught it.
    joined = "".join(c.text for c in chunks)
    ws_joined = "".join(joined.split())
    for sec in sections:
        sents = sec.split(". ")
        assert "".join(sents[0].split()) in ws_joined, "section head lost during chunking"
        assert "".join(sents[-1].split()) in ws_joined, "section tail lost during chunking"
    # Adjacent chunks share the bridge tail (modulo boundary whitespace).
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.text[: chunker.chunk_overlap - 2] in prev.text


def test_section_headers_start_chunks():
    """Structured docs: all-caps headers must head their own chunks."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=20, strategy="recursive")
    text = (
        "Jane Doe\njane@example.com\n\n"
        "EXPERIENCE\n"
        + ("Senior Engineer at Acme Corp — built systems.\n" * 12)
        + "SKILLS\nPython, SQL, Docker\n"
    )
    chunks = chunker._chunk_text(
        text, doc_id="d3", source="t.txt", doc_type="text", extra_metadata={}
    )
    texts = [c.text for c in chunks]
    assert any(t.startswith("EXPERIENCE") for t in texts), "EXPERIENCE header buried"
    assert any(t.startswith("SKILLS") for t in texts), "SKILLS header buried"


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