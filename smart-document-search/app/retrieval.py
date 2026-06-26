"""Hybrid chunk retrieval: dense embeddings + TF-IDF lexical search."""

from __future__ import annotations

from pathlib import Path

from app.chunking import chunk_documents
from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    HYBRID_TFIDF_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
)
from app.document_loader import load_txt_documents
from app.embeddings import EmbeddingModel
from app.preprocessing import preprocess_text
from app.search_engine import DocumentSearchEngine
from app.vector_store import VectorStore

# Retrieval strategy
# ------------------
# We use HYBRID search (not TF-IDF-only fallback):
#   • Embeddings (all-MiniLM-L6-v2): semantic similarity — paraphrases, concepts.
#   • TF-IDF: lexical keyword overlap — exact terms, acronyms, rare tokens.
# Scores from each method are min-max normalized within the candidate set, then
# combined: final = HYBRID_VECTOR_WEIGHT * vector + HYBRID_TFIDF_WEIGHT * tfidf.
# If the embedding model or ChromaDB fails at startup, retrieval falls back to
# TF-IDF over the same chunks (see HybridRetriever._vector_enabled).


class HybridRetriever:
    """Index document chunks and rank them with hybrid dense + sparse search."""

    def __init__(self) -> None:
        self._tfidf = DocumentSearchEngine()
        self._vector_store = VectorStore(CHROMA_DIR)
        self._embedder: EmbeddingModel | None = None
        self._vector_enabled = False
        self._chunks: list[dict[str, str]] = []

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def retrieval_method(self) -> str:
        if self._vector_enabled:
            return "Hybrid (Embeddings + TF-IDF)"
        return "TF-IDF (fallback)"

    def fit(self, chunks: list[dict[str, str]]) -> None:
        """Build TF-IDF and vector indexes from preprocessed chunks."""
        if not chunks:
            raise ValueError("No chunks to index.")

        self._chunks = chunks
        indexed = [
            {
                "name": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "content": chunk["content"],
                "processed_content": preprocess_text(chunk["content"]),
            }
            for chunk in chunks
        ]

        self._tfidf.fit(indexed)

        try:
            self._embedder = EmbeddingModel()
            self._vector_store.reset()
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self._embedder.encode(texts)
            self._vector_store.upsert_chunks(chunks, embeddings)
            self._vector_enabled = True
        except Exception:
            self._embedder = None
            self._vector_enabled = False

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        include_content: bool = False,
    ) -> list[dict[str, float | str]]:
        """Return top-k chunks with source file, chunk id, and combined score."""
        cleaned = preprocess_text(query)
        if not cleaned.strip():
            return []

        candidate_k = min(max(top_k * 4, top_k), len(self._chunks) or top_k)

        tfidf_hits = self._tfidf.search(
            query, top_k=candidate_k, include_content=True
        )
        for hit in tfidf_hits:
            hit.setdefault("source", hit.get("name", ""))

        if not self._vector_enabled or self._embedder is None:
            return self._finalize_hits(tfidf_hits[:top_k], include_content)

        query_vector = self._embedder.encode_query(query)
        vector_hits = self._vector_store.search(query_vector, top_k=candidate_k)

        merged = _merge_hybrid_scores(tfidf_hits, vector_hits)
        return self._finalize_hits(merged[:top_k], include_content)

    @staticmethod
    def _finalize_hits(
        hits: list[dict[str, float | str]],
        include_content: bool,
    ) -> list[dict[str, float | str]]:
        results: list[dict[str, float | str]] = []
        for hit in hits:
            if hit.get("score", 0) <= 0:
                continue
            item: dict[str, float | str] = {
                "chunk_id": str(hit.get("chunk_id", "")),
                "source": str(hit.get("source") or hit.get("name", "")),
                "name": str(hit.get("source") or hit.get("name", "")),
                "score": round(float(hit["score"]), 4),
            }
            if include_content:
                item["content"] = str(hit.get("content", ""))
            results.append(item)
        return results


def _normalize_scores(hits: list[dict[str, float | str]]) -> None:
    """Scale scores in-place to [0, 1] for fair hybrid weighting."""
    if not hits:
        return
    values = [float(hit["score"]) for hit in hits]
    low, high = min(values), max(values)
    if high == low:
        for hit in hits:
            hit["score"] = 1.0
        return
    for hit in hits:
        hit["score"] = (float(hit["score"]) - low) / (high - low)


def _merge_hybrid_scores(
    tfidf_hits: list[dict[str, float | str]],
    vector_hits: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Combine sparse and dense rankings by weighted normalized scores."""
    tfidf_copy = [dict(hit) for hit in tfidf_hits]
    vector_copy = [dict(hit) for hit in vector_hits]
    _normalize_scores(tfidf_copy)
    _normalize_scores(vector_copy)

    combined: dict[str, dict[str, float | str]] = {}

    for hit in tfidf_copy:
        chunk_id = str(hit["chunk_id"])
        combined[chunk_id] = {
            **hit,
            "score": HYBRID_TFIDF_WEIGHT * float(hit["score"]),
        }

    for hit in vector_copy:
        chunk_id = str(hit["chunk_id"])
        vector_score = HYBRID_VECTOR_WEIGHT * float(hit["score"])
        if chunk_id in combined:
            combined[chunk_id]["score"] = float(combined[chunk_id]["score"]) + vector_score
        else:
            combined[chunk_id] = {**hit, "score": vector_score}

    ranked = sorted(combined.values(), key=lambda h: float(h["score"]), reverse=True)
    for hit in ranked:
        hit["score"] = round(float(hit["score"]), 4)
    return ranked


def build_retriever(documents_dir: Path = DOCUMENTS_DIR) -> HybridRetriever:
    """Load documents, chunk them, and build hybrid indexes."""
    raw_docs = load_txt_documents(documents_dir)
    chunks = chunk_documents(
        raw_docs,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    retriever = HybridRetriever()
    retriever.fit(chunks)
    return retriever
