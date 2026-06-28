# Testplan — RAG Document Search System

Dieses Dokument beschreibt die automatisierte Teststrategie für das Backend (`smart-document-search/app/`). Ziel: reproduzierbare Qualitätssicherung für Retrieval, RAG und API — ohne echte LLM- oder Embedding-Modelle in CI.

## Testziele

| Ziel | Beschreibung |
|------|--------------|
| **Korrektheit** | Kernlogik liefert erwartete Ergebnisse bei bekannten Eingaben |
| **Robustheit** | Edge Cases (leere Eingaben, fehlende Indexe, Provider-Ausfall) werden abgefangen |
| **Regressionsschutz** | Jeder Push/PR löst die vollständige Testsuite aus (GitHub Actions) |
| **Nachvollziehbarkeit** | Coverage-Report zeigt getestete und ungetestete Codepfade |

## Testarten

### Unit Tests

Isolierte Tests einzelner Funktionen/Klassen mit Mocks für externe Abhängigkeiten.

| Modul | Testdatei | Was wird geprüft |
|-------|-----------|------------------|
| `preprocessing.py` | `test_preprocessing.py` | Normalisierung, Tokenisierung |
| `chunking.py` | `test_chunking.py` | Leerer Text, Overlap, Absatzgrenzen, Chunk-IDs |
| `search_engine.py` | `test_search_engine.py` | Index-Aufbau, leere Queries, Ranking |
| `retrieval.py` | `test_retrieval.py` | TF-IDF/Hybrid-Suche, Score-Normalisierung, Merge |
| `rag.py` | `test_rag.py` | Context-Building, Fallback, LLM-Mock, Provider-Logik |
| `embeddings.py` | `test_embeddings.py` | Leere Batches, Wrapper um SentenceTransformer (gemockt) |
| `vector_store.py` | `test_vector_store.py` | ChromaDB Upsert/Search/Reset |
| `document_loader.py` | `test_document_loader.py` | TXT/PDF-Laden, README-Skip, OCR-Pfade (gemockt) |

### Integration Tests

Tests über Modulgrenzen hinweg mit temporären Dateisystem-Corpora.

| Szenario | Testdatei | Beschreibung |
|----------|-----------|--------------|
| Corpus → Index → Suche | `test_search_engine.py` | `build_search_engine()` mit Temp-Ordner |
| Corpus → Chunks → Retriever | `test_retrieval.py` | `build_retriever()` End-to-End (TF-IDF-Fallback) |
| API → RAG → Antwort | `test_api.py` | FastAPI `TestClient` mit gemocktem LLM |

### API Tests

| Endpoint | Erwartetes Verhalten |
|----------|---------------------|
| `GET /api/health` | `ok` wenn Retriever geladen, sonst `degraded` |
| `GET /api/stats` | Dokument-/Chunk-Anzahl, Retrieval-Methode |
| `GET /search` | Ranked Chunks mit `chunk_id` und Score |
| `POST /api/ask` | Antwort + Quellen; 422 bei ungültigem Body |
| Fehlerpfade | 503 wenn Retriever nicht initialisiert |

## Was bewusst gemockt wird

| Abhängigkeit | Grund |
|--------------|-------|
| **LLM (Groq/OpenAI/Ollama)** | Keine API-Keys/Kosten in CI; `_try_llm` wird gemockt |
| **SentenceTransformer** | Kein Modell-Download in CI; Fake-Embeddings in Fixtures |
| **PDF/OCR (Tesseract, Poppler)** | Systemabhängig; Pfade werden per Monkeypatch simuliert |

## Fixtures (`conftest.py`)

- **`sample_documents`** — Zwei In-Memory-Dokumente (RAG + Docker)
- **`documents_dir`** — Temporärer Corpus-Ordner inkl. README-Skip
- **`tfidf_retriever`** — HybridRetriever im TF-IDF-Fallback-Modus
- **`hybrid_retriever`** — HybridRetriever mit Fake-Embeddings + echtem ChromaDB (Temp-Pfad)

## Ausführung

```bash
cd smart-document-search
pip install -r requirements.txt
python -m pytest
```

Coverage-Report (Terminal + HTML):

```bash
python -m pytest
# HTML-Report: htmlcov/index.html
```

Schwellwert: **≥ 80 %** (konfiguriert in `.coveragerc`).

## CI/CD

Workflow: `.github/workflows/ci.yml`

- Trigger: Push/PR auf `main`/`master`
- Python 3.12, Ubuntu
- `pytest` mit Coverage-Gate
- HTML-Coverage als Artifact (14 Tage)

## Nicht im Scope

- Frontend-Tests (`frontend/`) — separates UI-Projekt
- Last-/Performance-Tests
- E2E-Tests gegen echte Groq-API
- `main.py` CLI (aus Coverage ausgenommen)

## Traceability-Matrix (Anforderung → Test)

| Anforderung | Test |
|-------------|------|
| Dokumente korrekt laden | `test_load_documents_*` |
| README-Dateien überspringen | `test_load_documents_reads_corpus_and_skips_readme` |
| Text vor Index normalisieren | `test_preprocess_*` |
| Dokumente in Chunks teilen | `test_chunk_*` |
| TF-IDF-Ranking | `test_search_*` |
| Hybrid-Retrieval | `test_hybrid_retriever_*`, `test_merge_hybrid_scores_*` |
| RAG-Antwort mit Quellen | `test_ask_*`, `test_api_ask_*` |
| Fallback ohne LLM | `test_ask_falls_back_when_llm_unavailable` |
| API-Verfügbarkeit | `test_api_health_*`, `test_search_returns_503_*` |
