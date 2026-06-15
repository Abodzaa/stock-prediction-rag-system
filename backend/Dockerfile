# Backend image. Build context = project root (needs ./backend and ./src).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=10 \
    PIP_REQUIRE_HASHES=0 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# System deps for scientific wheels are bundled in slim manylinux wheels; keep minimal.
RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN grep -v "^torch" /app/backend/requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir --retries ${PIP_RETRIES} --timeout ${PIP_DEFAULT_TIMEOUT} -r /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir --retries ${PIP_RETRIES} --timeout ${PIP_DEFAULT_TIMEOUT} --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

# Application + research package (artifacts are mounted as a volume, see .dockerignore).
COPY backend/ /app/backend/
COPY src/ /app/src/

ENV RESEARCH_ROOT=/app/src \
    MODELS_DIR=/app/src/artifacts \
    MODEL_REGISTRY=/app/backend/app/model_registry.json \
    TORCH_DEVICE=cpu

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
