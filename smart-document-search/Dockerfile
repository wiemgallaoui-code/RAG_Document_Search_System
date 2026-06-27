FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=on \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip

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

RUN mkdir -p /app/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
