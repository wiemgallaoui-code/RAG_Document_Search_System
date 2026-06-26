"""Split documents into overlapping text chunks for retrieval."""

from __future__ import annotations

# Target 500–800 characters per chunk (configurable via config.py).
DEFAULT_CHUNK_SIZE = 650
DEFAULT_CHUNK_OVERLAP = 100


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Return overlapping chunks from ``text``, breaking on natural boundaries when possible."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer paragraph, line, or sentence breaks inside the second half of the window.
            search_from = start + chunk_size // 2
            for separator in ("\n\n", "\n", ". ", "? ", "! "):
                boundary = text.rfind(separator, search_from, end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, str]]:
    """Expand loaded documents into chunk records with stable ids and source metadata."""
    chunks: list[dict[str, str]] = []

    for doc in documents:
        source = doc["name"]
        for index, piece in enumerate(
            chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                {
                    "chunk_id": f"{source}:{index}",
                    "source": source,
                    "content": piece,
                }
            )

    return chunks
