# Smart Document Search

Work in progress: a small Python project for exploring document search over local text files. The repo currently provides layout, configuration, and sample inputs—**search is not implemented yet**.

## Repository layout

- **`app/`** — Application code (`main` entry point, `config` for paths; add modules such as ingest/search as you go).
- **`documents/`** — Plain-text sources (`.txt`) used for experiments; replace the samples with your own files as needed.
- **`requirements.txt`** — Project dependencies; install with `pip install -r requirements.txt` when you add packages.
- **`.gitignore`** — Excludes virtual environments, caches, local env files, and common IDE artifacts from version control.

## Run locally

From this directory (`smart-document-search`):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m app.main
```

On macOS/Linux, activate with `source .venv/bin/activate` instead of the PowerShell line above.

Expected output: a short status line plus the resolved path to `documents/`.

## Roadmap

- Implement ingestion and search under `app/`, then connect them from `main.py`.
- Pin dependency versions in `requirements.txt` once non-stdlib libraries are in use.
