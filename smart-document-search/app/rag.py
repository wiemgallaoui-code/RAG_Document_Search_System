"""Retrieval-augmented generation: hybrid chunk retrieval + LLM answer."""

from __future__ import annotations

import httpx

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MAX_CONTEXT_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from app.retrieval import HybridRetriever

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant answering questions from a document knowledge base. "
    "Use ONLY the provided document excerpts to answer. "
    "Write in a direct, natural, and helpful tone — as if you already know the material. "
    "Never say phrases like 'According to the provided context', 'Based on the documents', "
    "or 'The context mentions'. Just answer directly. "
    "Structure longer answers with short paragraphs, bullet points, or numbered lists when helpful. "
    "If the excerpts do not contain enough information, say what is missing in one sentence. "
    "Answer in the same language as the user's question. Be concise but complete."
)


def build_context(hits: list[dict[str, float | str]]) -> str:
    """Format retrieved documents into a single context string for the LLM."""
    parts: list[str] = []
    remaining = MAX_CONTEXT_CHARS

    for hit in hits:
        name = str(hit["name"])
        content = str(hit.get("content", "")).strip()
        if not content:
            continue

        header = f"--- {name} ({hit.get('chunk_id', '')}) ---\n"
        budget = remaining - len(header)
        if budget <= 0:
            break

        snippet = content if len(content) <= budget else content[: budget - 3] + "..."
        parts.append(header + snippet)
        remaining -= len(header) + len(snippet)

    return "\n\n".join(parts)


def _extractive_fallback(query: str, hits: list[dict[str, float | str]]) -> str:
    """Simple answer when no LLM is configured or the provider fails."""
    if not hits:
        return "No relevant documents found for your question."

    lines = [
        f"Based on the indexed documents, here are the most relevant excerpts for: {query}\n"
    ]
    for hit in hits:
        name = hit["name"]
        content = str(hit.get("content", "")).strip()
        preview = content[:300] + ("..." if len(content) > 300 else "")
        lines.append(f"[{name}] (score: {hit['score']})\n{preview}\n")

    lines.append(
        "Tip: set GROQ_API_KEY in .env (free at console.groq.com) for a generated answer."
    )
    return "\n".join(lines)


def _chat_completion(
    *,
    api_key: str,
    model: str,
    base_url: str | None,
    query: str,
    context: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _generate_groq(query: str, context: str) -> str:
    return _chat_completion(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        base_url="https://api.groq.com/openai/v1",
        query=query,
        context=context,
    )


def _generate_openai(query: str, context: str) -> str:
    return _chat_completion(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=None,
        query=query,
        context=context,
    )


def _generate_ollama(query: str, context: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        "stream": False,
    }
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    return data.get("message", {}).get("content", "")


def _try_llm(query: str, context: str) -> tuple[str, str] | None:
    """Return (answer, provider) or None if no provider succeeded."""
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        try:
            return _generate_groq(query, context), "groq"
        except Exception:
            pass

    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        try:
            return _generate_openai(query, context), "openai"
        except Exception:
            pass

    if LLM_PROVIDER == "ollama":
        try:
            return _generate_ollama(query, context), "ollama"
        except httpx.HTTPError:
            pass

    # Auto-fallback: try any configured provider when primary fails.
    if LLM_PROVIDER != "groq" and GROQ_API_KEY:
        try:
            return _generate_groq(query, context), "groq"
        except Exception:
            pass

    if LLM_PROVIDER != "openai" and OPENAI_API_KEY:
        try:
            return _generate_openai(query, context), "openai"
        except Exception:
            pass

    if LLM_PROVIDER != "ollama":
        try:
            return _generate_ollama(query, context), "ollama"
        except httpx.HTTPError:
            pass

    return None


def ask(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 3,
) -> dict[str, object]:
    """Run retrieval + generation and return a structured RAG response."""
    hits = retriever.search(query, top_k=top_k, include_content=True)

    if not hits:
        return {
            "query": query.strip(),
            "answer": "No relevant documents found for your question.",
            "sources": [],
            "provider": "none",
        }

    context = build_context(hits)
    llm_result = _try_llm(query, context)

    if llm_result:
        answer, provider = llm_result
    else:
        answer = _extractive_fallback(query, hits)
        provider = "fallback"

    sources = [
        {
            "document": hit.get("source") or hit["name"],
            "chunk_id": hit.get("chunk_id", ""),
            "similarity_score": hit["score"],
        }
        for hit in hits
    ]

    return {
        "query": query.strip(),
        "answer": answer,
        "sources": sources,
        "provider": provider,
    }
