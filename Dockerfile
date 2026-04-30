# YODEP — Multi-stage Docker build
# Stage 1: base system + Python deps
# Stage 2: runtime image with code

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — install Python packages into a clean prefix
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-venv \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.4 driver)
RUN pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining requirements
COPY requirements.txt /tmp/requirements.txt

# Strip torch lines (already installed above) before installing remainder
RUN grep -v "^torch" /tmp/requirements.txt | pip install --no-cache-dir -r /dev/stdin

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Deterministic CUDA
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    # HuggingFace cache inside container volume
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface" \
    # Experiment outputs
    YODEP_RESULTS="/app/results"

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN useradd -m -u 1000 yodep
WORKDIR /app

# Copy project code
COPY --chown=yodep:yodep . /app/

# Install project package in editable mode
RUN pip install -e /app --no-deps

# Create all necessary directories and set ownership
RUN mkdir -p \
        /app/data/yodep/raw \
        /app/data/yodep/processed \
        /app/data/yodep/features \
        /app/data/daic_woz/raw \
        /app/data/daic_woz/processed \
        /app/data/daic_woz/features \
        /app/results/tables \
        /app/results/figures \
        /app/logs \
        /app/.cache/huggingface \
        /app/.cache/features \
        /app/.cache/exp_ckpts \
    && chown -R yodep:yodep /app

USER yodep

# Default entrypoint runs the full experiment suite
# Override with docker run ... python experiments/run_yodep_main.py
ENTRYPOINT ["python"]
CMD ["experiments/run_all.py", "--skip-daic"]

# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="YODEP" \
      org.opencontainers.image.description="Yoruba-English Acted Depression Speech — F0 Transferability Study" \
      org.opencontainers.image.source="https://github.com/[YOUR_USERNAME]/yodep" \
      org.opencontainers.image.licenses="MIT"
