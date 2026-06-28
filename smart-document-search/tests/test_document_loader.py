"""Tests for document loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.document_loader import load_documents, load_txt_documents


def test_load_documents_reads_corpus_and_skips_readme(documents_dir: Path):
    docs = load_documents(documents_dir)

    names = {doc["name"] for doc in docs}
    assert names == {"rag_guide.txt", "docker.txt"}
    assert all(doc["content"].strip() for doc in docs)


def test_load_txt_documents_alias(documents_dir: Path):
    assert load_txt_documents(documents_dir) == load_documents(documents_dir)


def test_load_documents_missing_folder_raises(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_documents(missing)


def test_load_documents_non_directory_raises(tmp_path: Path):
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("content", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        load_documents(file_path)


def test_should_skip_readme_variants():
    from app.document_loader import _should_skip

    assert _should_skip(Path("README.txt")) is True
    assert _should_skip(Path("readme.pdf")) is True
    assert _should_skip(Path("notes.txt")) is False


def test_load_documents_reads_pdf(tmp_path: Path, monkeypatch):
    (tmp_path / "rag_guide.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "app.document_loader._read_pdf",
        lambda _path: (
            "Retrieval-Augmented Generation (RAG) combines information retrieval "
            "with text generation in PDF form."
        ),
    )

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0]["name"] == "rag_guide.pdf"
    assert "Retrieval-Augmented Generation" in docs[0]["content"]


def test_load_documents_skips_empty_pdf(tmp_path: Path, monkeypatch):
    (tmp_path / "empty.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setattr("app.document_loader._read_pdf", lambda _path: "   ")

    assert load_documents(tmp_path) == []


def test_read_pdf_falls_back_to_ocr_when_text_missing(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("app.document_loader._extract_pdf_text", lambda _path: "")
    monkeypatch.setattr(
        "app.document_loader._ocr_pdf",
        lambda _path: "Scanned document text recovered by OCR.",
    )

    from app.document_loader import _read_pdf

    assert _read_pdf(pdf_path) == "Scanned document text recovered by OCR."


def test_read_pdf_skips_ocr_when_disabled(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("app.document_loader.OCR_ENABLED", False)
    monkeypatch.setattr("app.document_loader._extract_pdf_text", lambda _path: "")

    def _ocr_should_not_run(_path):
        raise AssertionError("OCR should not run when disabled")

    monkeypatch.setattr("app.document_loader._ocr_pdf", _ocr_should_not_run)

    from app.document_loader import _read_pdf

    assert _read_pdf(pdf_path) == ""


def test_read_pdf_prefers_embedded_text_over_ocr(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "mixed.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "app.document_loader._extract_pdf_text",
        lambda _path: "Embedded PDF text layer.",
    )

    def _ocr_should_not_run(_path):
        raise AssertionError("OCR should not run when embedded text exists")

    monkeypatch.setattr("app.document_loader._ocr_pdf", _ocr_should_not_run)

    from app.document_loader import _read_pdf

    assert _read_pdf(pdf_path) == "Embedded PDF text layer."
