from app.config import DOCUMENTS_DIR
from app.retrieval import build_retriever


def main() -> None:
    print("Smart document search")
    print(f"Documents folder: {DOCUMENTS_DIR}\n")

    retriever = build_retriever()
    print(f"Indexed {retriever.chunk_count} chunk(s) — {retriever.retrieval_method}\n")
    query = "python package indexing"
    print(f"Example query: {query!r}\n")

    results = retriever.search(query, top_k=3)
    if not results:
        print("No matches found.")
        return

    print("Top results:")
    for hit in results:
        print(
            f"  - {hit['source']} [{hit['chunk_id']}]  (similarity: {hit['score']})"
        )


if __name__ == "__main__":
    main()
