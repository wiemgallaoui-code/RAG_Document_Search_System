"""TF-IDF indexing and cosine-similarity search over local documents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import DOCUMENTS_DIR
from app.document_loader import load_txt_documents
from app.preprocessing import preprocess_documents, preprocess_text


class DocumentSearchEngine:
    """Index preprocessed documents and rank them against a query."""

    def __init__(self) -> None:
        # Documents are already lowercased; avoid double-processing in the vectorizer.
        self._vectorizer = TfidfVectorizer(lowercase=False)
        self._documents: list[dict[str, str]] = []
        self._matrix = None

    def fit(self, processed_documents: list[dict[str, str]]) -> None:
        """Build the TF-IDF matrix from loaded, preprocessed documents."""
        if not processed_documents:
            raise ValueError("No documents to index.")

        self._documents = processed_documents
        corpus = [doc["processed_content"] for doc in processed_documents]

        # Step 1: turn each document into a TF-IDF vector.
        self._matrix = self._vectorizer.fit_transform(corpus) 

    def search(self, query: str, top_k: int = 3) -> list[dict[str, float | str]]:
        """Return the top_k documents most similar to the query."""
        if self._matrix is None:
            raise RuntimeError("Call fit() before search().")

        cleaned_query = preprocess_text(query)
        if not cleaned_query.strip():
            return []

        # Step 2: vectorize the query with the same vocabulary as the corpus.
        query_vector = self._vectorizer.transform([cleaned_query])

        # Step 3: cosine similarity — 1.0 means identical direction, 0.0 means no overlap.
        scores = cosine_similarity(query_vector, self._matrix).flatten()

        # Step 4: pick the highest scores (stable sort by score, then name).
        ranked_indices = np.argsort(scores)[::-1]
        results: list[dict[str, float | str]] = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])
            if score <= 0.0:
                continue
            doc = self._documents[index]
            results.append(
                {
                    "name": doc["name"],
                    "score": round(score, 4),
                }
            )

        return results


def build_search_engine(documents_dir: Path = DOCUMENTS_DIR) -> DocumentSearchEngine:
    """Load documents from disk, preprocess them, and build the index."""
    raw_docs = load_txt_documents(documents_dir)
    processed = preprocess_documents(raw_docs)
    engine = DocumentSearchEngine()
    engine.fit(processed)
    return engine


if __name__ == "__main__":
    engine = build_search_engine()
    demo_query = "python indexing search"
    print(f"Query: {demo_query!r}\n")

    for hit in engine.search(demo_query, top_k=3):
        print(f"  {hit['name']}  (similarity: {hit['score']})")
