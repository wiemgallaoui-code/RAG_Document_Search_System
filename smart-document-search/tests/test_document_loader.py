"""Tests for document loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.document_loader import load_txt_documents


def test_load_txt_documents_reads_corpus_and_skips_readme(documents_dir: Path):
    docs = load_txt_documents(documents_dir)

    names = {doc["name"] for doc in docs}
    assert names == {"rag_guide.txt", "docker.txt"}
    assert all(doc["content"].strip() for doc in docs)


def test_load_txt_documents_missing_folder_raises(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_txt_documents(missing)
