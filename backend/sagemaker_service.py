import os
import json
import logging
from typing import Dict, Any

import boto3
import botocore.exceptions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    def generate_response(self, prompt: str, max_length: int = 1024) -> Dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload)
            )

            body = response["Body"].read().decode("utf-8")
            data = json.loads(body)

            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text") or data[0].get("generated_texts")
            elif isinstance(data, dict):
                text = data.get("generated_text") or data.get("text") or data.get("output")
            else:
                text = None

            if not text:
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
