"""Shared pytest fixtures for the RAG document search project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.chunking import chunk_documents
from app.retrieval import HybridRetriever

SAMPLE_RAG_DOC = {
    "name": "rag_guide.txt",
    "content": (
        "Retrieval-Augmented Generation (RAG) combines information retrieval "
        "with text generation. RAG fetches relevant document chunks at query "
        "time and uses them as context for the LLM answer."
    ),
}

SAMPLE_DOCKER_DOC = {
    "name": "docker.txt",
    "content": (
        "Docker containers package applications with their dependencies. "
        "Images are built from Dockerfiles and run in isolated environments."
    ),
}


@pytest.fixture
def sample_documents() -> list[dict[str, str]]:
    return [SAMPLE_RAG_DOC, SAMPLE_DOCKER_DOC]


@pytest.fixture
def documents_dir(tmp_path: Path, sample_documents: list[dict[str, str]]) -> Path:
    """Temporary corpus folder with one searchable file and one README skip."""
    for doc in sample_documents:
        (tmp_path / doc["name"]).write_text(doc["content"], encoding="utf-8")
    (tmp_path / "README.txt").write_text("Not indexed", encoding="utf-8")
    return tmp_path


class FakeEmbeddingModel:
    """Deterministic embeddings for hybrid retrieval tests (no model download)."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._model_name = "fake-test-model"

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 8), dtype=np.float32)
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 1000
            vec = np.array([(seed + i) / 1000.0 for i in range(8)], dtype=np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            vectors.append(vec)
        return np.stack(vectors)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]


@pytest.fixture
def tfidf_retriever(tmp_path: Path, sample_documents: list[dict[str, str]], monkeypatch):
    """HybridRetriever using TF-IDF only (no embedding model load in tests)."""
    monkeypatch.setattr("app.retrieval.CHROMA_DIR", tmp_path / "chroma_test")

    def _embedding_unavailable(*_args, **_kwargs):
        raise RuntimeError("embeddings disabled in tests")

    monkeypatch.setattr("app.retrieval.EmbeddingModel", _embedding_unavailable)

    chunks = chunk_documents(sample_documents, chunk_size=120, overlap=20)
    retriever = HybridRetriever()
    retriever.fit(chunks)
    assert retriever.retrieval_method == "TF-IDF (fallback)"
    return retriever


@pytest.fixture
def hybrid_retriever(tmp_path: Path, sample_documents: list[dict[str, str]], monkeypatch):
    """HybridRetriever with mocked embeddings and real ChromaDB on a temp path."""
    monkeypatch.setattr("app.retrieval.CHROMA_DIR", tmp_path / "chroma_hybrid")
    monkeypatch.setattr("app.retrieval.EmbeddingModel", FakeEmbeddingModel)

    chunks = chunk_documents(sample_documents, chunk_size=120, overlap=20)
    retriever = HybridRetriever()
    retriever.fit(chunks)
    assert retriever.retrieval_method == "Hybrid (Embeddings + TF-IDF)"
    return retriever
