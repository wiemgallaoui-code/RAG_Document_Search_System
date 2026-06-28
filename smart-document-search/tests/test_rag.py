"""Tests for RAG answer generation."""

from __future__ import annotations

from app import rag
from app.rag import (
    _build_sources,
    _extractive_fallback,
    ask,
    build_context,
)
from app.retrieval import HybridRetriever


def test_build_context_formats_hits():
    hits = [
        {
            "name": "rag_guide.txt",
            "chunk_id": "rag_guide.txt:0",
            "content": "RAG combines retrieval with generation.",
        }
    ]
    context = build_context(hits)

    assert "rag_guide.txt" in context
    assert "RAG combines retrieval" in context


def test_build_context_truncates_at_max_chars(monkeypatch):
    monkeypatch.setattr("app.rag.MAX_CONTEXT_CHARS", 80)
    hits = [
        {
            "name": "long.txt",
            "chunk_id": "long.txt:0",
            "content": "x" * 200,
        },
        {
            "name": "second.txt",
            "chunk_id": "second.txt:0",
            "content": "should not appear",
        },
    ]

    context = build_context(hits)

    assert len(context) <= 80
    assert "should not appear" not in context


def test_build_context_skips_empty_content():
    hits = [
        {"name": "empty.txt", "chunk_id": "empty.txt:0", "content": "   "},
        {
            "name": "valid.txt",
            "chunk_id": "valid.txt:0",
            "content": "Valid excerpt.",
        },
    ]
    context = build_context(hits)

    assert "Valid excerpt." in context
    assert "empty.txt" not in context


def test_extractive_fallback_without_hits():
    answer = _extractive_fallback("What is RAG?", [])
    assert "No relevant documents" in answer


def test_extractive_fallback_lists_excerpts():
    hits = [
        {
            "name": "rag_guide.txt",
            "chunk_id": "rag_guide.txt:0",
            "score": 0.91,
            "content": "RAG combines retrieval with generation.",
        }
    ]
    answer = _extractive_fallback("What is RAG?", hits)

    assert "rag_guide.txt" in answer
    assert "RAG combines retrieval" in answer
    assert "GROQ_API_KEY" in answer


def test_build_sources_maps_hit_fields():
    hits = [
        {
            "source": "docker.txt",
            "name": "docker.txt",
            "chunk_id": "docker.txt:0",
            "score": 0.75,
        }
    ]
    sources = _build_sources(hits)

    assert sources[0]["document"] == "docker.txt"
    assert sources[0]["chunk_id"] == "docker.txt:0"
    assert sources[0]["similarity_score"] == 0.75


def test_ask_returns_none_provider_when_no_hits(tfidf_retriever: HybridRetriever, monkeypatch):
    monkeypatch.setattr(tfidf_retriever, "search", lambda *args, **kwargs: [])
    result = ask(tfidf_retriever, "anything", top_k=1)

    assert result["provider"] == "none"
    assert result["sources"] == []
    assert "No relevant documents" in result["answer"]


def test_ask_uses_mocked_llm(tfidf_retriever: HybridRetriever, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_try_llm",
        lambda _query, _context: ("Generated answer from LLM.", "groq"),
    )

    result = ask(tfidf_retriever, "What is Retrieval-Augmented Generation?", top_k=2)

    assert result["answer"] == "Generated answer from LLM."
    assert result["provider"] == "groq"
    assert len(result["sources"]) >= 1


def test_ask_falls_back_when_llm_unavailable(tfidf_retriever: HybridRetriever, monkeypatch):
    monkeypatch.setattr(rag, "_try_llm", lambda _query, _context: None)

    result = ask(tfidf_retriever, "What is RAG?", top_k=2)

    assert result["provider"] == "fallback"
    assert "rag_guide.txt" in result["answer"] or "RAG" in result["answer"]


def test_try_llm_uses_groq_when_configured(monkeypatch):
    monkeypatch.setattr(rag, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(rag, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        rag,
        "_generate_groq",
        lambda _query, _context: "Groq answer",
    )

    result = rag._try_llm("question", "context")

    assert result == ("Groq answer", "groq")


def test_try_llm_returns_none_when_all_providers_fail(monkeypatch):
    monkeypatch.setattr(rag, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(rag, "GROQ_API_KEY", "")
    monkeypatch.setattr(rag, "OPENAI_API_KEY", "")

    def _ollama_fails(*_args, **_kwargs):
        import httpx

        raise httpx.HTTPError("offline")

    monkeypatch.setattr(rag, "_generate_ollama", _ollama_fails)

    assert rag._try_llm("question", "context") is None
