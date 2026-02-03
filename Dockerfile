# EMG Gesture Recognition Application
# Multi-stage Docker build for production deployment

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# DEPENDENCIES STAGE
# =============================================================================
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# APPLICATION STAGE
# =============================================================================
FROM dependencies AS application

# Copy application code
COPY config.py .
COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY data/ ./data/

# Create models directory
RUN mkdir -p /app/models

# =============================================================================
# PRODUCTION STAGE
# =============================================================================
FROM application AS production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Run with gunicorn for production
# Note: Models are trained automatically on first startup if not present
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "app:app"]

# =============================================================================
# DEVELOPMENT STAGE
# =============================================================================
FROM application AS development

# Development mode with Flask's built-in server
CMD ["python", "app.py"]
