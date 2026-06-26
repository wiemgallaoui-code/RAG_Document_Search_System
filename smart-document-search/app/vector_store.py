"""ChromaDB-backed vector index for document chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    """Persist chunk embeddings in ChromaDB for cosine similarity search."""

    def __init__(
        self,
        persist_dir: Path,
        *,
        collection_name: str = "document_chunks",
    ) -> None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection (used when rebuilding the corpus index)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: list[dict[str, str]],
        embeddings: np.ndarray,
    ) -> None:
        """Index all chunks with precomputed embedding vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas: list[dict[str, Any]] = [
            {"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
            for chunk in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
    ) -> list[dict[str, float | str]]:
        """Return the closest chunks by cosine distance (converted to similarity)."""
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, float | str]] = []
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            # Chroma cosine distance = 1 - cosine_similarity.
            similarity = max(0.0, 1.0 - float(distance))
            hits.append(
                {
                    "chunk_id": str(metadata.get("chunk_id", chunk_id)),
                    "source": str(metadata.get("source", "")),
                    "name": str(metadata.get("source", "")),
                    "content": document or "",
                    "score": round(similarity, 4),
                }
            )

        return hits
