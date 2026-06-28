"""Tests for embedding model wrapper."""

from __future__ import annotations

import numpy as np


def test_encode_empty_list_returns_empty_array(monkeypatch):
    class FakeTransformer:
        def encode(self, *_args, **_kwargs):
            raise AssertionError("encode should not be called for empty input")

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda _name: FakeTransformer(),
    )

    from app.embeddings import EmbeddingModel

    model = EmbeddingModel("fake-model")
    result = model.encode([])

    assert result.shape == (0, 0)


def test_encode_returns_normalized_vectors(monkeypatch):
    class FakeTransformer:
        def encode(self, texts, **_kwargs):
            return np.array([[1.0, 2.0, 3.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda _name: FakeTransformer(),
    )

    from app.embeddings import EmbeddingModel

    model = EmbeddingModel("fake-model")
    vectors = model.encode(["hello", "world"])

    assert vectors.shape == (2, 3)
    assert model.model_name == "fake-model"


def test_encode_query_delegates_to_encode(monkeypatch):
    class FakeTransformer:
        def encode(self, texts, **_kwargs):
            return np.array([[0.1, 0.2]], dtype=np.float32)

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda _name: FakeTransformer(),
    )

    from app.embeddings import EmbeddingModel

    model = EmbeddingModel("fake-model")
    query_vector = model.encode_query("test query")

    assert query_vector.shape == (2,)
