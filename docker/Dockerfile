FROM python:3.9-slim

LABEL maintainer="rohan.vinaik@example.com"
LABEL description="Anti-Pattern-Mining Framework for Multi-Modal Subtype Discovery"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    vim \
    less \
    ca-certificates \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash researcher && \
    chown -R researcher:researcher /workspace

# Copy project files
COPY --chown=researcher:researcher pyproject.toml README.md LICENSE ./
COPY --chown=researcher:researcher src/ ./src/
COPY --chown=researcher:researcher configs/ ./configs/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir jupyter ipykernel ipywidgets

# Create data directory structure
RUN mkdir -p \
    data/raw \
    data/interim \
    data/processed \
    data/external \
    data/example \
    results/figures \
    results/benchmarks \
    results/reports \
    audit_logs \
    checkpoints \
    && chown -R researcher:researcher data results audit_logs checkpoints

# Copy example data
COPY --chown=researcher:researcher docker/example_data/ ./data/example/

# Copy scripts
COPY --chown=researcher:researcher scripts/ ./scripts/

# Switch to non-root user
USER researcher

# Set up Jupyter kernel
RUN python -m ipykernel install --user --name audhd --display-name "Python (AuDHD Pipeline)"

# Expose Jupyter port
EXPOSE 8888

# Set entrypoint
ENTRYPOINT ["python", "-m", "audhd_correlation.pipeline.main"]

# Default command shows help
CMD ["--help"]
