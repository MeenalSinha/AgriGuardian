# ==============================================================================
# AgriGuardian Multi-Stage Production Dockerfile (Phase 12)
# ==============================================================================

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Final production runtime stage
FROM python:3.11-slim AS runtime

# Create isolated, unprivileged application service user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install runtime curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

# Copy application source code
COPY --chown=appuser:appgroup . /app

# Create persistent storage directories with proper unprivileged ownership
RUN mkdir -p /app/data /app/logs /app/audio /app/reports /app/backups /app/temp && \
    chown -R appuser:appgroup /app

USER appuser

# Healthcheck monitoring the live FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

EXPOSE 8000 8501

# Default command launches FastAPI headless backend + PWA static server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

