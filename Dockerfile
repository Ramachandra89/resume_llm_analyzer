# Multi-stage build for the Resume Coach FastAPI backend.
# Stage 1 installs deps; Stage 2 copies only what's needed — keeps image small.

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

# Install everything except the local-model extras (torch is huge — not needed
# when the LLM runs on SageMaker)
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
      fastapi uvicorn pydantic \
      PyPDF2 requests boto3 python-dotenv \
      beautifulsoup4 lxml openai \
      streamlit

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY backend/   ./backend/
COPY frontend/  ./frontend/
COPY prompts/   ./prompts/
COPY inference_check/ ./inference_check/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# SAGEMAKER_ENDPOINT_NAME must be set in the environment (via --env-file or -e)
# When set, api.py automatically uses SageMakerService instead of LocalModelService
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
