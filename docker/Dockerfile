# Stage 1: Builder (Compiles dependencies and wheel packages)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies & Ubuntu 22.04 native Python dev packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up isolated virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy packaging requirements
COPY requirements.txt pyproject.toml ./

# Install PyTorch runtime and dependencies inside virtualenv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Production Minimal Runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/workspace" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Install minimal runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace

# Copy essential codebase files
COPY omnipeft ./omnipeft
COPY configs ./configs
COPY evaluate.py train.py ./

# Create default target mount directories
RUN mkdir -p /workspace/artifacts /workspace/data_storage /root/.cache/huggingface

ENTRYPOINT ["python3"]
CMD ["evaluate.py", "--base_model", "HuggingFaceTB/SmolLM2-135M", "--fused_model", "HuggingFaceTB/SmolLM2-135M"]