"""HTTP API for document search (FastAPI).

Run from the ``smart-document-search`` folder::

    uvicorn app.api:app --reload

Then open http://127.0.0.1:8000/search?q=your+terms
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from app.search_engine import DocumentSearchEngine, build_search_engine

_engine: DocumentSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and index documents once when the server starts."""
    global _engine
    _engine = build_search_engine()
    yield
    _engine = None


app = FastAPI(
    title="Smart Document Search",
    description="TF-IDF + cosine similarity over local .txt files.",
    lifespan=lifespan,
)


@app.get("/search")
def search(
    q: str = Query(..., description="Search terms"),
    top_k: int = Query(3, ge=1, le=50, description="Max number of hits"),
) -> dict[str, Any]:
    """Return the most relevant documents for ``q``."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    hits = _engine.search(q, top_k=top_k)
    top_results = [
        {"document": hit["name"], "similarity_score": hit["score"]}
        for hit in hits
    ]

    return {
        "query": q.strip(),
        "top_results": top_results,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Try GET /search?q=your+query"}
