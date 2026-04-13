"""
Modal deployment for the Resume Coach FastAPI backend.

The backend calls Nebius AI for all LLM inference — no GPU needed on Modal.
Modal here provides serverless hosting + auto-scaling for the API.

Deploy:
    modal deploy inference_check/modal_inference.py

After deploying, Modal prints the HTTPS URL for your backend, e.g.:
    https://your-workspace--resume-coach-api.modal.run

Set that URL as BACKEND_API_URL in your Streamlit app's environment
(or the existing frontend/app.py).

Requires a Modal secret named "resume-coach-secrets" containing:
    NEBIUS_API_KEY    — your Nebius AI Studio API key
    (optional) HUGGINGFACE_HUB_TOKEN
"""

import modal

# ---------------------------------------------------------------------------
# Image: install only what the FastAPI backend needs (no torch/GPU)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "openai>=1.30.0",
        "PyPDF2>=3.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "huggingface-hub>=0.18.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.24.0",
        "boto3>=1.28.0",
    )
    # Copy the entire project source into the container
    .add_local_dir(".", remote_path="/app")
)

app = modal.App("resume-coach-api", image=image)

secrets = modal.Secret.from_name("resume-coach-secrets", required=False)


@app.function(
    secrets=[secrets],
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    """
    Serves the existing backend/api.py FastAPI application on Modal.
    LLM calls inside the backend route to Nebius via NEBIUS_API_KEY.
    """
    import sys

    sys.path.insert(0, "/app")

    # Patch LocalModelService to use Nebius instead of a local model
    # so the existing api.py works without torch/transformers.
    from inference_check.nebius_service import NebiusService
    from backend import local_model_service as lms

    class NebiusBackedService:
        """Drop-in replacement for LocalModelService that calls Nebius."""

        def __init__(self):
            self._svc = NebiusService()  # reads NEBIUS_API_KEY from env

        def generate_response(self, prompt: str, max_length: int = 1024) -> dict:
            return self._svc.generate(prompt, max_tokens=max_length)

    # Monkey-patch so api.py picks up Nebius transparently
    lms.LocalModelService = NebiusBackedService

    from backend.api import app as fastapi_application

    return fastapi_application


# ---------------------------------------------------------------------------
# Local test:  modal run inference_check/modal_inference.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def test():
    import urllib.request
    import json

    # Just hit the health endpoint to confirm the app starts
    print("Deploy with:  modal deploy inference_check/modal_inference.py")
    print("Then test:    curl <your-modal-url>/health")
