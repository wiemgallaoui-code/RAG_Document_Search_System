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


@pytest.fixture
def client_no_retriever(tfidf_retriever: HybridRetriever, monkeypatch):
    monkeypatch.setattr("app.api.build_retriever", lambda documents_dir=None: tfidf_retriever)
    with TestClient(api.app) as test_client:
        monkeypatch.setattr(api, "_retriever", None)
        yield test_client


def test_api_health_returns_ok_when_retriever_ready(client: TestClient):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["retriever_ready"] is True


def test_api_health_degraded_when_retriever_missing(client_no_retriever: TestClient):
    response = client_no_retriever.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["retriever_ready"] is False


def test_api_stats_returns_corpus_info(client: TestClient):
    response = client.get("/api/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] >= 1
    assert data["chunk_count"] >= 2
    assert "TF-IDF" in data["retrieval_method"]
    assert data["llm_provider"]


def test_search_returns_ranked_chunks(client: TestClient):
    response = client.get("/search", params={"q": "Retrieval-Augmented Generation", "top_k": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["query"]
    assert len(data["top_results"]) >= 1
    hit = data["top_results"][0]
    assert hit["document"] == "rag_guide.txt"
    assert hit["chunk_id"].startswith("rag_guide.txt:")
    assert hit["similarity_score"] > 0


def test_search_returns_503_without_retriever(client_no_retriever: TestClient):
    response = client_no_retriever.get("/search", params={"q": "RAG"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Search engine not ready"


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


def test_api_ask_returns_503_without_retriever(client_no_retriever: TestClient):
    response = client_no_retriever.post(
        "/api/ask",
        json={"query": "What is RAG?", "top_k": 1},
    )

    assert response.status_code == 503


def test_api_ask_rejects_empty_query(client: TestClient):
    response = client.post("/api/ask", json={"query": "", "top_k": 3})

    assert response.status_code == 422


def test_api_ask_rejects_invalid_top_k(client: TestClient):
    response = client.post(
        "/api/ask",
        json={"query": "What is RAG?", "top_k": 0},
    )

    assert response.status_code == 422


def test_root_returns_json_or_html(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(("text/html", "application/json"))
