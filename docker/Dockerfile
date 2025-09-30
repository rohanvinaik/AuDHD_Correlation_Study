FROM python:3.10-slim

LABEL maintainer="your.email@example.com"
LABEL description="Multi-omics integration pipeline for ADHD/Autism subtyping"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy the rest of the project
COPY . .

# Create data directories
RUN mkdir -p data/raw data/interim data/processed data/external

# Default command
CMD ["audhd", "--help"]