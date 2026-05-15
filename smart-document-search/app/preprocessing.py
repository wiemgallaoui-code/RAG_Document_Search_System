"""Normalize raw document text before search indexing.

Why preprocess?
  Search compares words, not exact punctuation. Without cleanup,
  "Python" and "python" or "search," and "search" look like different
  terms. Lowercasing and removing punctuation helps TF-IDF match queries
  to documents more reliably.
"""

from __future__ import annotations

import string


def preprocess_text(text: str) -> str:
    """Return cleaned text as a single space-separated string of tokens."""
    # Same casing everywhere so "Machine" and "machine" count as one term.
    text = text.lower()

    # Turn punctuation into spaces: "hello, world" -> "hello  world"
    punctuation_map = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(punctuation_map)

    # Split on whitespace and drop empty pieces left after punctuation removal.
    tokens = [token for token in text.split() if token]

    return " ".join(tokens)


def preprocess_documents(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add a ``processed_content`` field to each loaded document dict."""
    result: list[dict[str, str]] = []
    for doc in documents:
        result.append(
            {
                "name": doc["name"],
                "content": doc["content"],
                "processed_content": preprocess_text(doc["content"]),
            }
        )
    return result


if __name__ == "__main__":
    sample = "Hello, World!  TF-IDF works with tokens — not punctuation."
    print("Before:", sample)
    print("After: ", preprocess_text(sample))
