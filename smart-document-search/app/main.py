from app.config import DOCUMENTS_DIR
from app.document_loader import load_txt_documents


def main() -> None:
    print("Smart document search — work in progress (search not wired yet).")
    print(f"Documents folder: {DOCUMENTS_DIR}\n")

    documents = load_txt_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['name']} ({len(doc['content'])} chars)")


if __name__ == "__main__":
    main()
