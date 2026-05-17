# Smart Document Search

Small document search demo: load `.txt` files from `documents/`, preprocess text, rank with **TF-IDF** and **cosine similarity** (scikit-learn). Optional **FastAPI** exposes search over HTTP.

## Layout

- **`app/`** — Loader, preprocessing, search engine, CLI (`main`), HTTP API (`api`)
- **`documents/`** — UTF-8 `.txt` corpus (sample files included)
- **`requirements.txt`** — Pinned dependencies

## Install

From `smart-document-search`:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux: `source .venv/bin/activate`

## CLI demo

```bash
python -m app.main
```

Or run the search module demo:

```bash
python -m app.search_engine
```

## Run the API

```bash
uvicorn app.api:app --reload
```

Server default: http://127.0.0.1:8000

### API usage

**Endpoint:** `GET /search`

| Parameter | Meaning |
|-----------|---------|
| `q` | Search query (required) |
| `top_k` | Max results (default `3`, max `50`) |

**Example request**

```http
GET http://127.0.0.1:8000/search?q=python+indexing&top_k=2
```

**Example response**

```json
{
  "query": "python indexing",
  "top_results": [
    {
      "document": "example_meeting_notes.txt",
      "similarity_score": 0.42
    }
  ]
}
```

Interactive docs: http://127.0.0.1:8000/docs

## Technologies

- Python 3
- FastAPI, Uvicorn
- scikit-learn (`TfidfVectorizer`, cosine similarity)

## Notes

- README-style `.txt` files under `documents/` named like `README.txt` are skipped by the loader.
- Empty or whitespace-only queries return `"top_results": []`.
