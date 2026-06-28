"""Tests for TF-IDF search engine."""

from __future__ import annotations

import pytest

from app.preprocessing import preprocess_documents
from app.search_engine import DocumentSearchEngine, build_search_engine


def test_fit_empty_documents_raises():
    engine = DocumentSearchEngine()
    with pytest.raises(ValueError, match="No documents"):
        engine.fit([])


def test_search_before_fit_raises():
    engine = DocumentSearchEngine()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        engine.search("query")


def test_search_empty_query_returns_empty():
    engine = DocumentSearchEngine()
    docs = preprocess_documents([{"name": "a.txt", "content": "hello world"}])
    engine.fit(docs)
    assert engine.search("   !!!") == []


def test_search_returns_ranked_hits(sample_documents):
    engine = DocumentSearchEngine()
    processed = preprocess_documents(sample_documents)
    engine.fit(processed)

    hits = engine.search("Retrieval-Augmented Generation RAG", top_k=2)

    assert len(hits) >= 1
    assert hits[0]["name"] == "rag_guide.txt"
    assert float(hits[0]["score"]) > 0


def test_search_include_content(sample_documents):
    engine = DocumentSearchEngine()
    processed = preprocess_documents(sample_documents)
    engine.fit(processed)

    hits = engine.search("Docker containers", top_k=1, include_content=True)

    assert hits[0]["content"]
    assert "Docker" in str(hits[0]["content"])


def test_build_search_engine_loads_corpus(documents_dir):
    engine = build_search_engine(documents_dir)
    hits = engine.search("Docker images", top_k=1)

    assert hits
    assert hits[0]["name"] == "docker.txt"
