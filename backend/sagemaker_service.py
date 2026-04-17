import os
import json
import logging
from typing import Dict, Any

import boto3
import botocore.exceptions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Active JumpStart endpoint — Llama 3.1 8B Instruct (generative text model).
LLAMA_31_8B_ENDPOINT = "jumpstart-dft-llama-3-1-8b-instruct-20260417-030131"


class SageMakerService:
    def __init__(self, endpoint_name: str = None, region_name: str = None):
        self.endpoint_name = endpoint_name or os.getenv("SAGEMAKER_ENDPOINT_NAME")
        self.region_name = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

        if not self.endpoint_name:
            raise ValueError("SAGEMAKER_ENDPOINT_NAME environment variable is required")
        if not self.region_name:
            raise ValueError("AWS_REGION or AWS_DEFAULT_REGION environment variable is required")

        self.client = boto3.client("sagemaker-runtime", region_name=self.region_name)
        logger.info(f"SageMaker endpoint set to {self.endpoint_name} in {self.region_name}")

    # ------------------------------------------------------------------
    # Llama Guard 3 8B — safety classification
    # ------------------------------------------------------------------
    def classify_safety(self, text: str) -> Dict[str, Any]:
        """Invoke the Llama Guard 3 8B endpoint to classify content safety.

        Returns a dict with:
            response  – "safe" | "unsafe\\n[category]"
            is_safe   – bool
            category  – violation category string, or None if safe
            status    – "success" | "error"
        """
        # Llama Guard expects a conversation-style prompt with the Llama 3
        # chat template.  The model outputs "safe" or "unsafe\n<S-code>".
        formatted = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Task: Check if there is unsafe content in the following message.\n\n"
            f"<message>\n{text}\n</message>\n\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        payload = {
            "inputs": formatted,
            "parameters": {
                "max_new_tokens": 20,
                "temperature": 0.01,
                "return_full_text": False,
            },
        }
        try:
            resp = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )
            raw = resp["Body"].read().decode("utf-8")
            data = json.loads(raw)
            text_out = self._extract_text(data)
            if text_out is None:
                raise ValueError(f"Unexpected response: {data}")
            text_out = text_out.strip()
            is_safe = text_out.lower().startswith("safe")
            category = None if is_safe else (text_out.split("\n")[1] if "\n" in text_out else text_out)
            return {"response": text_out, "is_safe": is_safe, "category": category, "status": "success"}
        except botocore.exceptions.ClientError as e:
            logger.error(f"SageMaker client error: {e}")
            return {"response": None, "is_safe": None, "category": None, "status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error invoking SageMaker endpoint: {e}")
            return {"response": None, "is_safe": None, "category": None, "status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Generic text generation (used by all /analyze, /chat, etc. routes)
    # ------------------------------------------------------------------
    def generate_response(self, prompt: str, max_length: int = 1024) -> Dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False,
            },
        }

        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )

            body = response["Body"].read().decode("utf-8")
            data = json.loads(body)
            text = self._extract_text(data)

            if text is None:
                raise ValueError(f"Unexpected SageMaker response format: {data}")

            if isinstance(text, list):
                text = text[0]

            return {"response": text, "status": "success"}

        except botocore.exceptions.ClientError as e:
            logger.error(f"SageMaker client error: {e}")
            return {"response": None, "status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error invoking SageMaker endpoint: {e}")
            return {"response": None, "status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text(data) -> str | None:
        """Parse TGI / SageMaker response shapes into a plain string."""
        if isinstance(data, list) and data:
            item = data[0]
            return item.get("generated_text") or item.get("generated_texts")
        if isinstance(data, dict):
            return data.get("generated_text") or data.get("text") or data.get("output")
        if isinstance(data, str):
            return data
        return None
