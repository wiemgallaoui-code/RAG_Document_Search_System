"""HTTP API for document search and RAG (FastAPI).

Run from the ``smart-document-search`` folder::

    uvicorn app.api:app --reload

Open http://127.0.0.1:8000 for the web UI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import DOCUMENTS_DIR, LLM_PROVIDER, OPENAI_API_KEY, GROQ_API_KEY, STATIC_DIR
from app.document_loader import load_txt_documents
from app.rag import ask as rag_ask
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
    title="RAG Document Assistant",
    description="TF-IDF retrieval with RAG answer generation (Groq / OpenAI / Ollama).",
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(3, ge=1, le=50, description="Number of source documents")


def _active_provider_label() -> str:
    """Human-readable label for the configured LLM provider."""
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return "Groq"
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return "OpenAI"
    if LLM_PROVIDER == "ollama":
        return "Ollama"
    if GROQ_API_KEY:
        return "Groq"
    if OPENAI_API_KEY:
        return "OpenAI"
    return "Fallback"


@app.get("/api/stats")
def stats() -> dict[str, str | int]:
    """Return corpus and system info for the frontend header."""
    docs = load_txt_documents(DOCUMENTS_DIR)
    return {
        "document_count": len(docs),
        "retrieval_method": "TF-IDF",
        "llm_provider": _active_provider_label(),
    }


@app.get("/search")
def search(
    q: str = Query(..., description="Search terms"),
    top_k: int = Query(3, ge=1, le=50, description="Max number of hits"),
) -> dict[str, Any]:
    """Return the most relevant documents for ``q`` (retrieval only)."""
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


@app.post("/api/ask")
def ask_endpoint(body: AskRequest) -> dict[str, Any]:
    """Retrieve relevant documents and generate a natural-language answer."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    return rag_ask(_engine, body.query, top_k=body.top_k)


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "Try GET /search?q=your+query or POST /api/ask"}
