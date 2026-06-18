"""ChromaDB-backed vector store wrapper.

Provides a thin, opinionated facade over ChromaDB with:

* idempotent collection creation
* batch upserts with metadata
* similarity search returning rich ``RetrievalResult`` objects
* persistence toggling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """One retrieved chunk with its score and metadata."""

    chunk_id: str
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)


class VectorStore:
    """ChromaDB-backed store for chunk embeddings."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./data/chroma",
        distance_metric: str = "cosine",
        persist: bool = True,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory if persist else None
        self.distance_metric = distance_metric

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError("chromadb not installed. Run `pip install chromadb`.") from exc

        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            settings = Settings(persist_directory=self.persist_directory, anonymized_telemetry=False)
            self.client = chromadb.PersistentClient(path=self.persist_directory, settings=settings)
        else:
            self.client = chromadb.EphemeralClient()

        # Distance metric mapping
        metric_map = {"cosine": "cosine", "l2": "l2", "ip": "ip"}
        metadata_config = {"hnsw:space": metric_map.get(distance_metric, "cosine")}

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata_config,
        )
        logger.info(
            "VectorStore ready (collection=%s, metric=%s, count=%d)",
            collection_name, distance_metric, self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        if not ids:
            return
        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids and embeddings must have the same length")

        # Chroma requires flat metadata; ensure no list values.
        clean_meta: List[Dict] = []
        for i, m in enumerate(metadatas or [{}] * len(ids)):
            flat = self._flatten_metadata(m or {})
            flat["_id"] = ids[i]
            clean_meta.append(flat)

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=clean_meta,
        )
        logger.info("Added %d vectors to collection '%s'", len(ids), self.collection_name)

    def upsert(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Insert or update existing IDs."""
        if not ids:
            return
        clean_meta: List[Dict] = []
        for i, m in enumerate(metadatas or [{}] * len(ids)):
            flat = self._flatten_metadata(m or {})
            flat["_id"] = ids[i]
            clean_meta.append(flat)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=clean_meta,
        )
        logger.info("Upserted %d vectors to collection '%s'", len(ids), self.collection_name)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """Return the top-k most similar chunks."""
        if self.collection.count() == 0:
            logger.warning("Query against empty collection")
            return []

        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, self.collection.count()),
        }
        if where:
            kwargs["where"] = where

        res = self.collection.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        # Chroma returns distances; convert to similarity score for cosine.
        distances = (res.get("distances") or [[]])[0]

        results: List[RetrievalResult] = []
        for i, cid in enumerate(ids):
            score = self._distance_to_score(distances[i]) if i < len(distances) else 0.0
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    text=docs[i] if i < len(docs) else "",
                    score=score,
                    metadata=metas[i] if i < len(metas) else {},
                )
            )
        return results

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop all data in the collection (destructive)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
        logger.warning("VectorStore collection '%s' reset", self.collection_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_metadata(meta: Dict, prefix: str = "") -> Dict:
        """Chroma only supports scalar metadata values. Convert everything
        to strings, drop empty values, and namespace keys with a prefix."""
        out: Dict[str, str] = {}
        for k, v in meta.items():
            if v is None:
                continue
            key = f"{prefix}{k}"
            if isinstance(v, (str, int, float, bool)):
                out[key] = str(v)
            else:
                out[key] = str(v)
        return out

    def _distance_to_score(self, distance: float) -> float:
        if self.distance_metric == "cosine":
            # cosine distance = 1 - similarity
            return max(0.0, min(1.0, 1.0 - float(distance)))
        if self.distance_metric == "ip":
            # inner product distance = -<a,b>
            return float(-distance)
        # l2
        return float(1.0 / (1.0 + distance))