"""
deploy_endpoint.py
------------------
Run this notebook cell-by-cell (or as a script) inside a SageMaker Studio
terminal / notebook to deploy Meta-Llama-3.1-8B-Instruct via JumpStart.

Prerequisites (run once in your Studio terminal):
    pip install "sagemaker>=2.200.0"

Usage:
    python deploy/deploy_endpoint.py [--instance-type ml.g5.2xlarge] [--endpoint-name resume-coach-endpoint]
"""

import argparse
import time

import boto3
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel

# ── Config ────────────────────────────────────────────────────────────────────
JUMPSTART_MODEL_ID = "meta-textgeneration-llama-3-1-8b-instruct"  # JumpStart ID
DEFAULT_ENDPOINT_NAME = "resume-coach-endpoint"
# ml.g5.2xlarge  → 1×A10G, 24 GB VRAM — fits 8B in fp16, ~$1.52/hr
# ml.g4dn.xlarge → 1×T4,  16 GB VRAM — needs 8-bit quant,   ~$0.74/hr
DEFAULT_INSTANCE_TYPE = "ml.g5.2xlarge"


def get_execution_role() -> str:
    """Return the SageMaker execution role for the current Studio session."""
    try:
        return sagemaker.get_execution_role()
    except Exception:
        # Fallback when running outside Studio (e.g. local with IAM user creds)
        sts = boto3.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        return f"arn:aws:iam::{account_id}:role/sagemaker-resume-coach"


def deploy(instance_type: str, endpoint_name: str) -> str:
    """
    Deploy Llama-3.1-8B-Instruct via SageMaker JumpStart.
    Returns the endpoint name once InService.
    """
    role = get_execution_role()
    session = sagemaker.Session()
    region = session.boto_region_name

    print(f"Region      : {region}")
    print(f"Role        : {role}")
    print(f"Model ID    : {JUMPSTART_MODEL_ID}")
    print(f"Instance    : {instance_type}")
    print(f"Endpoint    : {endpoint_name}")
    print()

    # Accept the Llama EULA — you must have already accepted it in AWS console
    # under SageMaker > JumpStart > Meta Llama 3.1 8B Instruct
    model = JumpStartModel(
        model_id=JUMPSTART_MODEL_ID,
        role=role,
        sagemaker_session=session,
    )

    print("Deploying endpoint (this takes ~8-12 minutes)…")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        # Accept Llama community license programmatically
        accept_eula=True,
    )

    print(f"\nEndpoint '{endpoint_name}' is InService.")
    print(f"\nAdd this to your .env file:\n")
    print(f"  SAGEMAKER_ENDPOINT_NAME={endpoint_name}")
    print(f"  AWS_REGION={region}")

    return predictor.endpoint_name


def wait_for_endpoint(endpoint_name: str, region: str = None) -> None:
    """Poll until the endpoint is InService (useful when deploying async)."""
    sm = boto3.client("sagemaker", region_name=region)
    print(f"Waiting for '{endpoint_name}' to be InService…", end="", flush=True)
    while True:
        status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if status == "InService":
            print(" done.")
            return
        if status in ("Failed", "OutOfService"):
            raise RuntimeError(f"Endpoint entered status: {status}")
        print(".", end="", flush=True)
        time.sleep(30)


def quick_smoke_test(endpoint_name: str, region: str = None) -> None:
    """Send a single test prompt to verify the endpoint works."""
    import json
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    payload = {
        "inputs": "What is machine learning?",
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }

    print("Running smoke test…")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    if isinstance(result, list):
        text = result[0].get("generated_text", "")
    else:
        text = result.get("generated_text", str(result))

    print("Model response:")
    print(text.strip())
    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME)
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Skip deployment; just test an existing endpoint",
    )
    args = parser.parse_args()

    if args.smoke_test_only:
        sm = boto3.client("sagemaker")
        region = sm.meta.region_name
        quick_smoke_test(args.endpoint_name, region)
    else:
        deploy(args.instance_type, args.endpoint_name)
        sm = boto3.client("sagemaker")
        quick_smoke_test(args.endpoint_name, sm.meta.region_name)
