"""Tests for hybrid retrieval and score merging."""

from __future__ import annotations

import pytest

from app.retrieval import (
    HybridRetriever,
    _merge_hybrid_scores,
    _normalize_scores,
    build_retriever,
)


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


def test_fit_empty_chunks_raises():
    retriever = HybridRetriever()
    with pytest.raises(ValueError, match="No chunks"):
        retriever.fit([])


def test_search_empty_query_returns_empty(tfidf_retriever: HybridRetriever):
    assert tfidf_retriever.search("   !!!") == []


def test_normalize_scores_equal_values():
    hits = [{"score": 0.5}, {"score": 0.5}]
    _normalize_scores(hits)
    assert hits[0]["score"] == 1.0
    assert hits[1]["score"] == 1.0


def test_normalize_scores_scales_to_unit_interval():
    hits = [{"score": 1.0}, {"score": 3.0}]
    _normalize_scores(hits)
    assert hits[0]["score"] == 0.0
    assert hits[1]["score"] == 1.0


def test_normalize_scores_empty_list_is_noop():
    hits: list[dict[str, float | str]] = []
    _normalize_scores(hits)
    assert hits == []


def test_merge_hybrid_scores_combines_tfidf_and_vector():
    tfidf_hits = [
        {
            "chunk_id": "doc.txt:0",
            "source": "doc.txt",
            "name": "doc.txt",
            "content": "alpha",
            "score": 0.8,
        }
    ]
    vector_hits = [
        {
            "chunk_id": "doc.txt:0",
            "source": "doc.txt",
            "name": "doc.txt",
            "content": "alpha",
            "score": 0.6,
        },
        {
            "chunk_id": "other.txt:0",
            "source": "other.txt",
            "name": "other.txt",
            "content": "beta",
            "score": 0.9,
        },
    ]

    merged = _merge_hybrid_scores(tfidf_hits, vector_hits)

    assert len(merged) == 2
    chunk_ids = {hit["chunk_id"] for hit in merged}
    assert chunk_ids == {"doc.txt:0", "other.txt:0"}
    assert float(merged[0]["score"]) >= float(merged[1]["score"])


def test_hybrid_retriever_uses_embeddings(hybrid_retriever: HybridRetriever):
    assert hybrid_retriever.retrieval_method == "Hybrid (Embeddings + TF-IDF)"
    hits = hybrid_retriever.search("RAG retrieval generation", top_k=2)
    assert hits
    assert all(float(hit["score"]) > 0 for hit in hits)


def test_build_retriever_end_to_end(documents_dir, monkeypatch, tmp_path):
    monkeypatch.setattr("app.retrieval.CHROMA_DIR", tmp_path / "chroma_build")

    def _embedding_unavailable(*_args, **_kwargs):
        raise RuntimeError("skip embeddings in integration test")

    monkeypatch.setattr("app.retrieval.EmbeddingModel", _embedding_unavailable)

    retriever = build_retriever(documents_dir)
    assert retriever.chunk_count >= 2
    hits = retriever.search("Docker containers", top_k=1)
    assert hits[0]["source"] == "docker.txt"
