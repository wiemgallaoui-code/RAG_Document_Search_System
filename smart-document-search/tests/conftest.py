"""Shared pytest fixtures for the RAG document search project."""

from __future__ import annotations

from pathlib import Path

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
