# Multi-stage Dockerfile for cross-platform testing
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY test_cross_platform.py .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Test stage
FROM base as test

# Run tests
RUN python test_cross_platform.py

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash agent_user
USER agent_user

# Default command
CMD ["python", "-m", "agent_cli", "--help"]
