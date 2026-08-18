# =============================================================================
# Dockerfile — Denial Prediction API (port 8000)
# =============================================================================
#
# Service  : model2main.py (FastAPI + XGBoost v2 + SHAP TreeExplainer)
# Port     : 8000
# Serves   : POST /predict_user_claim   — XGBoost inference + SHAP explanation
#            GET  /health               — liveness + model + SHAP status
#            GET  /model-info           — feature contract + global importance
#            GET  /api                  — service info
#            /    (static)              — frontend SPA (StaticFiles mount)
#
# Build:
#   docker build -t denial-prediction-api .
#
# Run (standalone):
#   docker run -p 8000:8000 denial-prediction-api
#
# Run (with docker-compose — see integration.txt section 10):
#   docker compose up prediction-api
# =============================================================================

FROM python:3.11-slim

# --------------------------------------------------------------------------
# System dependencies
# libgomp1  — required by XGBoost and LightGBM (OpenMP threading)
# libpq-dev — required to compile psycopg2 C extension
# gcc       — C compiler needed for psycopg2 build
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Working directory
# --------------------------------------------------------------------------
WORKDIR /app

# --------------------------------------------------------------------------
# Python dependencies
# Install before copying source so Docker caches this layer when only
# source files change.
# --------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        "fastapi>=0.140.0" \
        "uvicorn>=0.52.0" \
        "joblib>=1.5.0" \
        "xgboost>=3.2.0" \
        "scikit-learn==1.6.1" \
        "numpy>=2.4.0" \
        "pandas>=3.0.0" \
        "pydantic>=2.13.0" \
        "python-dotenv>=1.2.0" \
        "shap>=0.51.0"

# --------------------------------------------------------------------------
# Application source
# model2main.py is the prediction API entry point.
# preprocessing/ and validators/ contain the pipeline helpers.
# models/ contains the serialised XGBoost artifact.
# frontend/ is served as a StaticFiles mount at /.
# --------------------------------------------------------------------------
COPY model2main.py           ./
COPY preprocessing/          ./preprocessing/
COPY validators/             ./validators/
COPY models/                 ./models/
COPY frontend/               ./frontend/

# --------------------------------------------------------------------------
# Environment defaults (override via --env-file or docker-compose env_file)
# --------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --------------------------------------------------------------------------
# Expose prediction API port
# --------------------------------------------------------------------------
EXPOSE 8000

# --------------------------------------------------------------------------
# Start the prediction API
# model2main:app  →  model2main.py, app = FastAPI(...)
# --------------------------------------------------------------------------
CMD ["uvicorn", "model2main:app", "--host", "0.0.0.0", "--port", "8000"]
