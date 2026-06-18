"""End-to-end Multi-Modal RAG pipeline.

Wires ingestion → processing → retrieval → generation into a single
``RAGPipeline`` object. The same object is reused for both the
Streamlit UI and batch/CLI use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .generation.llm_interface import LLMInterface
from .ingestion.image_ocr import ImageOCR
from .ingestion.pdf_parser import PDFParser
from .ingestion.text_loader import TextLoader
from .processing.chunker import Chunk, TextChunker
from .processing.embedder import Embedder
from .retrieval.retriever import Retriever
from .retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class RAGPipeline:
    """All-in-one RAG orchestrator."""

    def __init__(self, config):
        cfg = config  # dot-notation Config object

        # Ingestion
        self.pdf_parser = PDFParser(
            max_pages=cfg.ingestion.pdf.max_pages,
            library=cfg.ingestion.pdf.library,
        )
        self.image_ocr = ImageOCR(
            engine=cfg.ingestion.image.ocr_engine,
            languages=list(cfg.ingestion.image.languages),
            preprocess=bool(cfg.ingestion.image.preprocess),
            min_confidence=int(cfg.ingestion.image.min_confidence),
        )
        self.text_loader = TextLoader(
            encoding=cfg.ingestion.text.encoding,
            extensions=list(cfg.ingestion.text.extensions),
        )

        # Processing
        self.chunker = TextChunker(
            chunk_size=int(cfg.processing.chunker.chunk_size),
            chunk_overlap=int(cfg.processing.chunker.chunk_overlap),
            strategy=str(cfg.processing.chunker.strategy),
            separators=list(cfg.processing.chunker.separators),
        )
        self.embedder = Embedder(
            model_name=str(cfg.processing.embedder.model_name),
            device=str(cfg.processing.embedder.device),
            normalize=bool(cfg.processing.embedder.normalize),
            batch_size=int(cfg.processing.embedder.batch_size),
            use_cache=bool(cfg.processing.embedder.cache),
        )

        # Retrieval
        self.vector_store = VectorStore(
            collection_name=str(cfg.retrieval.vector_store.collection_name),
            persist_directory=str(cfg.paths.vector_store_dir),
            distance_metric=str(cfg.retrieval.vector_store.distance_metric),
            persist=bool(cfg.retrieval.vector_store.persist),
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=int(cfg.retrieval.retriever.top_k),
            score_threshold=float(cfg.retrieval.retriever.score_threshold),
        )

        # Generation
        self.llm = LLMInterface(
            provider=str(cfg.generation.llm.provider),
            model_name=str(cfg.generation.llm.model_name),
            max_new_tokens=int(cfg.generation.llm.max_new_tokens),
            temperature=float(cfg.generation.llm.temperature),
            device=str(cfg.generation.llm.device),
            prompt_template=str(cfg.generation.prompt.template),
        )

        self.config = cfg
        logger.info("RAGPipeline initialised")

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest_file(self, file_path: PathLike) -> List[Chunk]:
        """Ingest a single file and return its chunks (also indexed)."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            doc = self.pdf_parser.parse(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            doc = self.image_ocr.extract(file_path)
            # Normalize structure for chunker
            doc = {
                "source": doc["source"],
                "type": "image",
                "full_text": doc["text"],
                "metadata": {
                    "ocr_confidence": doc.get("confidence", 0.0),
                    "num_words": len(doc.get("words", [])),
                },
            }
        else:
            doc = self.text_loader.load(file_path)

        chunks = self.chunker.chunk_document(doc)
        self.retriever.index_chunks(chunks)
        return chunks

    def ingest_files(self, file_paths: List[PathLike]) -> int:
        total = 0
        for fp in file_paths:
            try:
                n = len(self.ingest_file(fp))
                total += n
                logger.info("Ingested %s: %d chunks", fp, n)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to ingest %s: %s", fp, exc)
        return total

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
    ) -> Dict:
        """Run a full RAG query: retrieve → generate."""
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            return {
                "question": question,
                "answer": "I don't have any indexed documents to search. Please upload files first.",
                "sources": [],
            }

        answer = self.llm.generate(
            question=question,
            context_chunks=[r.text for r in results],
        )

        sources: List[Dict] = []
        if return_sources:
            for r in results:
                src = {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "text": r.text[:300] + ("..." if len(r.text) > 300 else ""),
                    "metadata": r.metadata,
                }
                sources.append(src)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        return {
            "num_chunks": self.vector_store.count(),
            "embedder_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "embedder_cache_size": self.embedder.cache_size,
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model_name,
        }