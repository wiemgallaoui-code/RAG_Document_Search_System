FROM node:22-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci 2>/dev/null || npm install

COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=on \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (~190MB) — no official pytorch/pytorch CPU image on Docker Hub
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries 10 \
    "torch==2.5.1+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries 10 \
    -r requirements-docker.txt

COPY app/ app/
COPY documents/ documents/
COPY static/ static/
COPY --from=frontend-build /frontend/dist frontend/dist/

RUN mkdir -p /app/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
