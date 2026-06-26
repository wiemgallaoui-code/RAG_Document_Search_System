"""Tests for text preprocessing."""

from __future__ import annotations

from app.preprocessing import preprocess_documents, preprocess_text


def test_preprocess_text_normalizes_case_and_punctuation():
    raw = "Hello, World!  TF-IDF works with tokens - not punctuation."
    cleaned = preprocess_text(raw)

    assert cleaned == "hello world tf idf works with tokens not punctuation"
    assert "," not in cleaned
    assert cleaned.islower()


def test_preprocess_documents_adds_processed_content():
    docs = [{"name": "a.txt", "content": "Machine Learning, RAG."}]
    result = preprocess_documents(docs)

    assert len(result) == 1
    assert result[0]["name"] == "a.txt"
    assert result[0]["processed_content"] == "machine learning rag"
