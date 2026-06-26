"""Tests for chunk retrieval."""

from __future__ import annotations

from app.retrieval import HybridRetriever


def test_retrieval_returns_relevant_chunk_for_known_query(tfidf_retriever: HybridRetriever):
    hits = tfidf_retriever.search(
        "Retrieval-Augmented Generation RAG",
        top_k=1,
    )

    assert len(hits) == 1
    top = hits[0]
    assert top["source"] == "rag_guide.txt"
    assert str(top["chunk_id"]).startswith("rag_guide.txt:")
    assert float(top["score"]) > 0
