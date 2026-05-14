"""Paths and simple settings for the application."""

from pathlib import Path

# Directory that contains this file → parent is project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
