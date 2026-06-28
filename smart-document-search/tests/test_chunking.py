"""Tests for document chunking."""

from __future__ import annotations

from app.chunking import chunk_documents, chunk_text


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "Short document text."
    assert chunk_text(text, chunk_size=100, overlap=10) == [text]


def test_chunk_text_splits_long_text_with_overlap():
    text = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50
    chunks = chunk_text(text, chunk_size=60, overlap=10)

    assert len(chunks) >= 2
    combined = " ".join(chunks)
    assert "A" in combined and "C" in combined


def test_chunk_text_prefers_paragraph_boundary():
    text = ("First paragraph content here.\n\n" + "Second paragraph " + "x" * 80)
    chunks = chunk_text(text, chunk_size=50, overlap=5)

    assert len(chunks) >= 2
    assert chunks[0].startswith("First paragraph")


def test_chunk_documents_assigns_stable_ids(sample_documents):
    chunks = chunk_documents(sample_documents, chunk_size=200, overlap=20)

    assert len(chunks) >= 2
    ids = {chunk["chunk_id"] for chunk in chunks}
    assert "rag_guide.txt:0" in ids
    assert all(chunk["source"] for chunk in chunks)
    assert all(chunk["content"].strip() for chunk in chunks)
