# =============================================================================
# Dockerfile — AI Analyst Microservice (port 8060)
# =============================================================================
#
# Service  : ai_analyst/main.py (FastAPI + Gemini via Vertex AI)
# Port     : 8060
# Serves   : POST /analyst/session     — create conversational session
#            POST /analyst/chat        — send message, receive LLM answer
#            DELETE /analyst/session/{id} — clean up session
#            GET  /health              — liveness + LLM status
#
# External dependencies:
#   - PostgreSQL  (session / message / claim-intelligence persistence)
#   - Google Vertex AI / Gemini  (LLM reasoning)
#
# Build:
#   docker build -t denial-analyst -f ai_analyst/Dockerfile .
#   (run from the PROJECT ROOT so the build context includes credentials.json)
#
# Run (standalone):
#   docker run -p 8060:8060 --env-file ai_analyst/.env denial-analyst
#
# Run (with docker-compose — see docker-compose.yml):
#   docker compose up analyst
# =============================================================================

FROM python:3.11-slim

# --------------------------------------------------------------------------
# System dependencies
# libpq-dev + gcc — required to build psycopg2 C extension
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Working directory
# --------------------------------------------------------------------------
WORKDIR /app

# --------------------------------------------------------------------------
# Python dependencies — install before copying source for layer caching
# --------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic \
        python-dotenv \
        sqlalchemy \
        psycopg2-binary \
        google-auth \
        google-genai

# --------------------------------------------------------------------------
# Application source
# Copy the entire ai_analyst package plus the GCP credentials file.
# credentials.json is expected in the project root.
# In production replace this with a mounted secret or Workload Identity.
# --------------------------------------------------------------------------
COPY ai_analyst/   ./ai_analyst/
COPY credentials.json ./credentials.json

# --------------------------------------------------------------------------
# Environment defaults
# GOOGLE_APPLICATION_CREDENTIALS must point to the credentials file inside
# the container.  Override DATABASE_URL and GOOGLE_CLOUD_PROJECT at runtime.
# --------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

# --------------------------------------------------------------------------
# Expose analyst service port
# --------------------------------------------------------------------------
EXPOSE 8060

# --------------------------------------------------------------------------
# Start the AI Analyst microservice
# The package is ai_analyst, the FastAPI app object is in ai_analyst/main.py
# --------------------------------------------------------------------------
CMD ["uvicorn", "ai_analyst.main:app", "--host", "0.0.0.0", "--port", "8060"]
