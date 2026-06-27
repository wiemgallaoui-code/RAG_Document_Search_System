"""Load plain-text and PDF documents from the documents folder."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def _should_skip(path: Path) -> bool:
    """Skip README-style instruction files (any supported extension)."""
    return path.stem.upper().startswith("README")


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def load_documents(documents_dir: Path) -> list[dict[str, str]]:
    """Read all .txt and .pdf files in documents_dir.

    Returns a list of dicts with:
      - name: file name (e.g. "notes.txt" or "report.pdf")
      - content: extracted text as a string

    Files named README.* are skipped (folder instructions, not search content).
    Empty files and PDFs with no extractable text are skipped.
    """
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {documents_dir}")
    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {documents_dir}")

    documents: list[dict[str, str]] = []

    for path in sorted(documents_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if _should_skip(path):
            continue

        if suffix == ".txt":
            content = _read_txt(path)
        else:
            content = _read_pdf(path)

        if not content.strip():
            continue

        documents.append({"name": path.name, "content": content})

    return documents


def load_txt_documents(documents_dir: Path) -> list[dict[str, str]]:
    """Backward-compatible alias for :func:`load_documents`."""
    return load_documents(documents_dir)


if __name__ == "__main__":
    from app.config import DOCUMENTS_DIR

    loaded = load_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(loaded)} file(s) from {DOCUMENTS_DIR}:\n")
    for doc in loaded:
        lines = doc["content"].strip().splitlines()
        first_line = lines[0] if lines else "(empty)"
        print(f"  {doc['name']}")
        print(f"    {first_line[:72]}{'...' if len(first_line) > 72 else ''}\n")
