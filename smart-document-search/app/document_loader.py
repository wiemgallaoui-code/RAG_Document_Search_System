"""Load plain-text documents from the documents folder."""

from __future__ import annotations

from pathlib import Path


def load_txt_documents(documents_dir: Path) -> list[dict[str, str]]:
    """Read all .txt files in documents_dir.

    Returns a list of dicts with:
      - name: file name (e.g. "notes.txt")
      - content: full file text as a string

    Files named README.txt are skipped (folder instructions, not search content).
    """
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {documents_dir}")
    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {documents_dir}")

    documents: list[dict[str, str]] = []

    for path in sorted(documents_dir.glob("*.txt")):
        if not path.is_file():
            continue
        # Skip README-style files so only real corpus files are loaded.
        if path.name.upper().startswith("README"):
            continue

        text = path.read_text(encoding="utf-8")
        documents.append({"name": path.name, "content": text})

    return documents


if __name__ == "__main__":
    from app.config import DOCUMENTS_DIR

    loaded = load_txt_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(loaded)} file(s) from {DOCUMENTS_DIR}:\n")
    for doc in loaded:
        lines = doc["content"].strip().splitlines()
        first_line = lines[0] if lines else "(empty)"
        print(f"  {doc['name']}")
        print(f"    {first_line[:72]}{'...' if len(first_line) > 72 else ''}\n")
