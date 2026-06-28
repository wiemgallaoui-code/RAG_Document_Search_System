"""Tests for ChromaDB vector store."""

from __future__ import annotations

import numpy as np
import pytest

from app.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(tmp_path / "chroma_vectors")


def test_upsert_and_search_returns_similar_chunks(vector_store: VectorStore):
    chunks = [
        {
            "chunk_id": "alpha.txt:0",
            "source": "alpha.txt",
            "content": "Retrieval augmented generation pipeline.",
        },
        {
            "chunk_id": "beta.txt:0",
            "source": "beta.txt",
            "content": "Docker container orchestration basics.",
        },
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    vector_store.upsert_chunks(chunks, embeddings)
    hits = vector_store.search(embeddings[0], top_k=2)

    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "alpha.txt:0"
    assert float(hits[0]["score"]) >= float(hits[1]["score"])


def test_search_empty_collection_returns_empty(vector_store: VectorStore):
    query = np.ones(3, dtype=np.float32)
    assert vector_store.search(query) == []


def test_upsert_length_mismatch_raises(vector_store: VectorStore):
    chunks = [{"chunk_id": "a:0", "source": "a.txt", "content": "text"}]
    embeddings = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="length mismatch"):
        vector_store.upsert_chunks(chunks, embeddings)


def test_reset_clears_collection(vector_store: VectorStore):
    chunks = [{"chunk_id": "a:0", "source": "a.txt", "content": "sample"}]
    embeddings = np.ones((1, 4), dtype=np.float32)

    vector_store.upsert_chunks(chunks, embeddings)
    vector_store.reset()

    assert vector_store.search(embeddings[0]) == []


def test_upsert_empty_chunks_is_noop(vector_store: VectorStore):
    vector_store.upsert_chunks([], np.empty((0, 4), dtype=np.float32))
    assert vector_store.search(np.ones(4, dtype=np.float32)) == []
