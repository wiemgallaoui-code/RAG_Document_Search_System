from app.config import DOCUMENTS_DIR
from app.search_engine import build_search_engine


def main() -> None:
    print("Smart document search")
    print(f"Documents folder: {DOCUMENTS_DIR}\n")

    engine = build_search_engine()
    query = "python package indexing"
    print(f"Example query: {query!r}\n")

    results = engine.search(query, top_k=3)
    if not results:
        print("No matches found.")
        return

    print("Top results:")
    for hit in results:
        print(f"  - {hit['name']}  (similarity: {hit['score']})")


if __name__ == "__main__":
    main()
