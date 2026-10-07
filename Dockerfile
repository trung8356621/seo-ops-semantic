FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/hf \
    FASTEMBED_CACHE_PATH=/models/fastembed

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts
COPY tests/fixtures ./tests/fixtures

RUN pip install --upgrade pip \
    && pip install .

RUN useradd --create-home --uid 10001 semantic \
    && mkdir -p /models \
    && chown -R semantic:semantic /app /models

USER semantic

EXPOSE 8088

# Single worker: avoid loading the ONNX model into multiple processes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088", "--workers", "1"]
