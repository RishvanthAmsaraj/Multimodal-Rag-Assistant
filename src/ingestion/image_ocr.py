"""Image OCR ingestion.

Wraps pytesseract (default) with an optional easyocr fallback.
Includes light image preprocessing for better OCR accuracy on
noisy inputs: grayscale conversion + contrast enhancement.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path, bytes]


class ImageOCR:
    """OCR extractor for JPG/PNG image files."""

    def __init__(
        self,
        engine: str = "pytesseract",
        languages: List[str] | None = None,
        preprocess: bool = True,
        min_confidence: int = 30,
    ):
        self.engine = engine
        self.languages = languages or ["eng"]
        self.preprocess = preprocess
        self.min_confidence = min_confidence

        # Defer backend init until first extract() call so the module
        # can be constructed in environments where the OCR binary
        # (or its python wrapper) isn't installed. This keeps the
        # rest of the pipeline usable for text-only workflows.
        self._backend_ready = False
        self._init_engine()

    def _init_engine(self) -> None:
        if self._backend_ready:
            return
        if self.engine == "pytesseract":
            try:
                self._check_pytesseract()
                self._backend_ready = True
            except ImportError:
                # Allow construction; first extract() will raise.
                self._backend_ready = False
        elif self.engine == "easyocr":
            try:
                self._init_easyocr()
                self._backend_ready = True
            except ImportError:
                self._backend_ready = False
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, image: PathLike) -> Dict:
        """Run OCR on an image.

        Parameters
        ----------
        image : path or bytes
            Path to an image file, or raw image bytes.

        Returns
        -------
        dict
            {
              "source": "...",
              "type": "image",
              "text": "...",
              "confidence": float,   # average word confidence, 0-100
              "words": [{text, conf, bbox}, ...]
            }
        """
        pil_img = self._load_image(image)
        pil_img = self._maybe_preprocess(pil_img)

        if self.engine == "pytesseract":
            return self._run_pytesseract(pil_img, image)
        return self._run_easyocr(pil_img, image)

    def extract_batch(self, images: List[PathLike]) -> List[Dict]:
        return [self.extract(img) for img in images]

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _run_pytesseract(self, pil_img, image: PathLike) -> Dict:
        if not self._backend_ready:
            self._init_engine()
        import pytesseract
        from pytesseract import Output

        # Full text + per-word detail
        lang_str = "+".join(self.languages)
        full_text = pytesseract.image_to_string(pil_img, lang=lang_str)

        data = pytesseract.image_to_data(pil_img, lang=lang_str, output_type=Output.DICT)
        words = []
        confs: List[float] = []
        for i, w in enumerate(data.get("text", [])):
            w = w.strip()
            if not w:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if conf >= self.min_confidence:
                words.append(
                    {
                        "text": w,
                        "conf": conf,
                        "bbox": {
                            "x": data.get("left", [0])[i],
                            "y": data.get("top", [0])[i],
                            "w": data.get("width", [0])[i],
                            "h": data.get("height", [0])[i],
                        },
                    }
                )
                confs.append(conf)

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        source = str(image) if isinstance(image, (str, Path)) else "<bytes>"

        logger.info("OCR (%s) extracted %d words from %s (avg conf %.1f)",
                    self.engine, len(words), source, avg_conf)

        return {
            "source": source,
            "type": "image",
            "text": full_text.strip(),
            "confidence": avg_conf,
            "words": words,
        }

    def _run_easyocr(self, pil_img, image: PathLike) -> Dict:
        if not getattr(self, "_backend_ready", False):
            self._init_engine()
        if not hasattr(self, "_reader"):
            self._init_easyocr()
        import numpy as np

        result = self._reader.readtext(np.array(pil_img))
        words: List[Dict] = []
        confs: List[float] = []
        text_parts: List[str] = []
        for bbox, text, conf in result:
            text = text.strip()
            if not text:
                continue
            if conf * 100 < self.min_confidence:
                continue
            words.append({"text": text, "conf": conf * 100, "bbox": {"corners": bbox}})
            confs.append(conf * 100)
            text_parts.append(text)

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        full_text = " ".join(text_parts)
        source = str(image) if isinstance(image, (str, Path)) else "<bytes>"

        logger.info("OCR (easyocr) extracted %d words from %s (avg conf %.1f)",
                    len(words), source, avg_conf)

        return {
            "source": source,
            "type": "image",
            "text": full_text.strip(),
            "confidence": avg_conf,
            "words": words,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_pytesseract() -> None:
        try:
            import pytesseract  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pytesseract not installed. Run `pip install pytesseract` and "
                "ensure the Tesseract binary is on PATH."
            ) from exc

    def _init_easyocr(self) -> None:
        try:
            import easyocr  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "easyocr not installed. Run `pip install easyocr` or set engine='pytesseract'."
            ) from exc
        import easyocr

        self._reader = easyocr.Reader(self.languages, gpu=False)

    @staticmethod
    def _load_image(image: PathLike):
        from PIL import Image

        if isinstance(image, (str, Path)):
            return Image.open(image)
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        raise TypeError(f"Unsupported image input: {type(image)}")

    def _maybe_preprocess(self, img):
        """Convert to grayscale + apply contrast enhancement."""
        if not self.preprocess:
            return img

        from PIL import Image, ImageEnhance, ImageOps

        # Grayscale
        if img.mode != "L":
            img = img.convert("L")

        # Auto-contrast for better OCR on uneven lighting
        img = ImageOps.autocontrast(img, cutoff=2)
        # Mild sharpening
        img = ImageEnhance.Contrast(img).enhance(1.3)
        return img