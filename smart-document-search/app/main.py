from app.config import DOCUMENTS_DIR
from app.document_loader import load_txt_documents
from app.preprocessing import preprocess_documents


def main() -> None:
    print("Smart document search — work in progress (search not wired yet).")
    print(f"Documents folder: {DOCUMENTS_DIR}\n")

    documents = load_txt_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['name']} ({len(doc['content'])} chars)")

    processed = preprocess_documents(documents)
    print("\nAfter preprocessing (first 80 chars of each):")
    for doc in processed:
        preview = doc["processed_content"][:80]
        suffix = "..." if len(doc["processed_content"]) > 80 else ""
        print(f"  - {doc['name']}: {preview}{suffix}")


if __name__ == "__main__":
    main()
