"""
test_endpoint.py
----------------
End-to-end test suite for the SageMaker endpoint and the FastAPI backend.
Run from SageMaker Studio terminal or locally (with AWS creds in env).

Usage:
    # Test endpoint directly (no FastAPI needed):
    python deploy/test_endpoint.py --mode endpoint

    # Test the FastAPI backend (must be running on localhost:8000):
    python deploy/test_endpoint.py --mode api

    # Both:
    python deploy/test_endpoint.py --mode all
"""

import argparse
import json
import os
import sys
import textwrap
import time

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "jumpstart-dft-llama-3-1-8b-instruct-20260417-030131",
)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

SAMPLE_RESUME = textwrap.dedent("""\
    Machine Learning Engineer with 5 years of experience.
    Skills: Python, PyTorch, TensorFlow, AWS, Docker, Kubernetes, FastAPI.
    Led deployment of NLP models serving 1M+ requests/day on AWS SageMaker.
    Built RAG pipeline reducing support ticket resolution time by 40%.
    MS Computer Science, Stanford University.
""")

SAMPLE_JD = textwrap.dedent("""\
    Senior ML Engineer — Responsibilities: design and deploy LLMs,
    build data pipelines, optimize inference latency, work with AWS.
    Requirements: Python, PyTorch, AWS, 3+ years ML experience.
""")

# ── Endpoint direct tests ─────────────────────────────────────────────────────

def test_endpoint_health() -> bool:
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    ok = status == "InService"
    _result("Endpoint InService", ok, f"status={status}")
    return ok


def test_endpoint_inference() -> bool:
    """Generic inference smoke-test.

    Works for both text-generation endpoints and Llama Guard 3 8B.
    Llama Guard returns 'safe' or 'unsafe\\n[category]' — very short — so we
    only require a non-empty string rather than > 5 characters.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    payload = {
        "inputs": "Machine learning is a subset of artificial intelligence.",
        "parameters": {"max_new_tokens": 20, "temperature": 0.1, "return_full_text": False},
    }
    t0 = time.time()
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    latency = time.time() - t0
    data = json.loads(resp["Body"].read())
    text = (data[0].get("generated_text") if isinstance(data, list) else data.get("generated_text", ""))
    ok = bool(text and text.strip())
    _result("Endpoint inference", ok, f"latency={latency:.1f}s  text='{text[:80]}'")
    return ok


def test_endpoint_llama_guard() -> bool:
    """Test Llama Guard 3 8B safety classification directly.

    Sends a known-safe message and expects the model to reply 'safe'.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    safe_text = "Please tell me the best way to improve my Python skills."
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Task: Check if there is unsafe content in the following message.\n\n"
        f"<message>\n{safe_text}\n</message>\n\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    payload = {
        "inputs": formatted,
        "parameters": {"max_new_tokens": 20, "temperature": 0.01, "return_full_text": False},
    }
    t0 = time.time()
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    latency = time.time() - t0
    data = json.loads(resp["Body"].read())
    text = (data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")).strip()
    ok = bool(text)
    _result(
        "Llama Guard safety classify",
        ok,
        f"latency={latency:.1f}s  result='{text[:80]}'",
    )
    return ok


# ── FastAPI backend tests ─────────────────────────────────────────────────────

def test_api_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        ok = r.status_code == 200
        _result("API /health", ok, r.json())
    except Exception as e:
        _result("API /health", False, str(e))
        ok = False
    return ok


def test_api_analyze() -> bool:
    payload = {"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD}
    try:
        r = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=300)
        ok = r.status_code == 200 and "analysis" in r.json()
        snippet = r.json().get("analysis", "")[:120] if ok else r.text[:120]
        _result("API /analyze", ok, snippet)
    except Exception as e:
        _result("API /analyze", False, str(e))
        ok = False
    return ok


def test_api_chat() -> bool:
    payload = {"conversation": "User: What skills should I highlight for an ML engineer role?\nCoach:"}
    try:
        r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
        ok = r.status_code == 200 and "response" in r.json()
        snippet = r.json().get("response", "")[:120] if ok else r.text[:120]
        _result("API /chat", ok, snippet)
    except Exception as e:
        _result("API /chat", False, str(e))
        ok = False
    return ok


def test_api_skill_assessment() -> bool:
    payload = {"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD}
    try:
        r = requests.post(f"{BACKEND_URL}/skill-assessment", json=payload, timeout=120)
        ok = r.status_code == 200 and "skill_match" in r.json()
        info = str(r.json().get("skill_match", {}).get("overall_score", "?")) if ok else r.text[:80]
        _result("API /skill-assessment", ok, f"overall_score={info}")
    except Exception as e:
        _result("API /skill-assessment", False, str(e))
        ok = False
    return ok


def test_api_scout_jobs() -> bool:
    payload = {"company_name": "Stripe", "resume_text": SAMPLE_RESUME}
    try:
        r = requests.post(f"{BACKEND_URL}/scout-jobs", json=payload, timeout=90)
        ok = r.status_code == 200 and "jobs" in r.json()
        info = f"{r.json().get('total_found')} jobs found" if ok else r.text[:80]
        _result("API /scout-jobs", ok, info)
    except Exception as e:
        _result("API /scout-jobs", False, str(e))
        ok = False
    return ok


def test_api_tailor() -> bool:
    payload = {
        "resume_text": SAMPLE_RESUME,
        "job_title": "Senior ML Engineer",
        "job_url": "https://stripe.com/jobs/listing/senior-ml-engineer/123",
        "job_description": SAMPLE_JD,
        "company_name": "Stripe",
    }
    try:
        r = requests.post(f"{BACKEND_URL}/tailor-for-job", json=payload, timeout=300)
        ok = r.status_code == 200 and "result" in r.json()
        snippet = r.json().get("result", "")[:120] if ok else r.text[:120]
        _result("API /tailor-for-job", ok, snippet)
    except Exception as e:
        _result("API /tailor-for-job", False, str(e))
        ok = False
    return ok


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result(name: str, ok: bool, detail) -> None:
    status = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  [{status}] {name:<35} {detail}")


def run_endpoint_tests() -> int:
    print("\n=== Direct endpoint tests ===")
    results = [test_endpoint_health(), test_endpoint_inference(), test_endpoint_llama_guard()]
    return sum(1 for r in results if not r)


def run_api_tests() -> int:
    print("\n=== FastAPI backend tests ===")
    results = [
        test_api_health(),
        test_api_analyze(),
        test_api_chat(),
        test_api_skill_assessment(),
        test_api_scout_jobs(),
        test_api_tailor(),
    ]
    return sum(1 for r in results if not r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["endpoint", "api", "all"], default="all")
    args = parser.parse_args()

    failures = 0
    if args.mode in ("endpoint", "all"):
        failures += run_endpoint_tests()
    if args.mode in ("api", "all"):
        failures += run_api_tests()

    print(f"\n{'=' * 50}")
    if failures == 0:
        print("\033[32mAll tests passed.\033[0m")
    else:
        print(f"\033[31m{failures} test(s) failed.\033[0m")
    sys.exit(failures)
