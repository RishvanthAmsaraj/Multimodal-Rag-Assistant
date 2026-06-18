"""Tests for ingestion modules.

These avoid hitting Tesseract / PyPDF in CI by mocking where
necessary, but still verify the real behavior on plain text
ingestion and OCR text shaping.
"""

from pathlib import Path

import pytest

from src.ingestion.image_ocr import ImageOCR
from src.ingestion.text_loader import TextLoader


def test_text_loader_reads_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello, RAG world!\nLine two.")
    loader = TextLoader()
    doc = loader.load(p)
    assert doc["type"] == "text"
    assert "Hello, RAG world!" in doc["text"]
    assert doc["metadata"]["size_bytes"] > 0


def test_text_loader_unsupported_extension(tmp_path):
    p = tmp_path / "sample.bin"
    p.write_bytes(b"binary data")
    loader = TextLoader()
    with pytest.raises(ValueError):
        loader.load(p)


def test_text_loader_missing_file(tmp_path):
    loader = TextLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "nope.txt")


def test_image_ocr_loads_bytes():
    """We don't have a Tesseract binary in CI, so we just verify the
    bytes-loading path doesn't crash before reaching the engine call."""
    from PIL import Image
    import io

    img = Image.new("RGB", (60, 30), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    # Pre-process check: should not raise
    pil = ImageOCR._load_image(raw_bytes)
    assert pil.size == (60, 30)
    # Construct OCR with a dummy engine so we can test preprocessing
    # without needing Tesseract installed.
    class _Stub:
        preprocess = True

    processed = ImageOCR._maybe_preprocess(_Stub(), pil)
    assert processed.mode == "L"


def test_pdf_parser_handles_missing_file(tmp_path):
    from src.ingestion.pdf_parser import PDFParser

    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.pdf")