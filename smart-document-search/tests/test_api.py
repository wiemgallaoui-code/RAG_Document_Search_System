"""Tests for FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import api
from app.retrieval import HybridRetriever


@pytest.fixture
def client(tfidf_retriever: HybridRetriever, monkeypatch):
    monkeypatch.setattr("app.api.build_retriever", lambda documents_dir=None: tfidf_retriever)
    with TestClient(api.app) as test_client:
        yield test_client


def test_api_ask_returns_200_with_mock_llm(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        "app.rag._try_llm",
        lambda _query, _context: ("Mocked LLM answer for testing.", "groq"),
    )

    response = client.post(
        "/api/ask",
        json={"query": "What is Retrieval-Augmented Generation?", "top_k": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked LLM answer for testing."
    assert data["provider"] == "groq"
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["document"] == "rag_guide.txt"
    assert "chunk_id" in data["sources"][0]
    assert "similarity_score" in data["sources"][0]
