"""Local sentence embeddings for semantic search (no API key required)."""

from __future__ import annotations

import numpy as np

from app.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Thin wrapper around sentence-transformers for document and query vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts; returns shape (n, dim)."""
        if not texts:
            return np.empty((0, 0))

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        return self.encode([query])[0]
