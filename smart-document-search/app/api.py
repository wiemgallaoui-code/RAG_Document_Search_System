"""HTTP API for document search and RAG (FastAPI).

Run from the ``smart-document-search`` folder::

    uvicorn app.api:app --reload

Open http://127.0.0.1:8000 for the web UI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import DOCUMENTS_DIR, LLM_PROVIDER, OPENAI_API_KEY, GROQ_API_KEY, STATIC_DIR
from app.document_loader import load_txt_documents
from app.rag import ask as rag_ask
from app.retrieval import HybridRetriever, build_retriever

_retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and index documents once when the server starts."""
    global _retriever
    _retriever = build_retriever()
    yield
    _retriever = None


app = FastAPI(
    title="RAG Document Assistant",
    description="Hybrid chunk retrieval (embeddings + TF-IDF) with RAG answer generation.",
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(3, ge=1, le=50, description="Number of source chunks")


class SourceHit(BaseModel):
    document: str = Field(..., description="Source file name")
    chunk_id: str = Field(..., description="Chunk identifier within the document")
    similarity_score: float = Field(..., description="Hybrid retrieval score")


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceHit]
    provider: Literal["groq", "openai", "ollama", "fallback", "none"]


class SearchHit(BaseModel):
    document: str
    chunk_id: str
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    top_results: list[SearchHit]


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
    method = _retriever.retrieval_method if _retriever else "Hybrid (Embeddings + TF-IDF)"
    return {
        "document_count": len(docs),
        "chunk_count": _retriever.chunk_count if _retriever else 0,
        "retrieval_method": method,
        "llm_provider": _active_provider_label(),
    }


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search terms"),
    top_k: int = Query(3, ge=1, le=50, description="Max number of hits"),
) -> SearchResponse:
    """Return the most relevant chunks for ``q`` (retrieval only)."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    hits = _retriever.search(q, top_k=top_k)
    top_results = [
        SearchHit(
            document=str(hit["source"]),
            chunk_id=str(hit["chunk_id"]),
            similarity_score=float(hit["score"]),
        )
        for hit in hits
    ]

    return SearchResponse(query=q.strip(), top_results=top_results)


@app.post("/api/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    """Retrieve relevant chunks and generate a natural-language answer."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    result = rag_ask(_retriever, body.query, top_k=body.top_k)
    return AskResponse(**result)


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "Try GET /search?q=your+query or POST /api/ask"}
