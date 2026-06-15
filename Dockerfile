FROM python:3.14-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for layer caching
COPY pyproject.toml ./
COPY src/ src/

# CPU-only Torch (fallback when LOW_MEMORY=false; FastEmbed is default on Render).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Keep BLAS/tokenizer memory low on constrained instances.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface

# Install the package (prod only - no dev extras like pytest/ruff)
RUN pip install --no-cache-dir .

# Pre-download FastEmbed ONNX model at build time (avoids HF fetch + OOM on first upload).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Copy remaining files
COPY . .

# Create upload + cache directories
RUN mkdir -p uploads /app/.cache/fastembed /app/.cache/huggingface

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "uvicorn rag_qa.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]