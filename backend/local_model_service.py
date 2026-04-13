import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
import botocore.exceptions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login as hf_login

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LocalModelService:
    def __init__(
        self,
        model_local_path: Optional[str] = None,
        hf_model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        device: Optional[str] = None,
        load_in_8bit: Optional[bool] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ):
        self.model_local_path = Path(model_local_path or os.getenv("MODEL_LOCAL_PATH", "./model"))
        self.hf_model_id = hf_model_id or os.getenv("HUGGINGFACE_MODEL_ID")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.s3_bucket = s3_bucket or os.getenv("S3_MODEL_BUCKET")
        self.s3_prefix = s3_prefix or os.getenv("S3_MODEL_PREFIX")
        self.device = (device or os.getenv("MODEL_DEVICE", "auto")).lower()
        self.load_in_8bit = (
            load_in_8bit
            if isinstance(load_in_8bit, bool)
            else str(os.getenv("LOAD_IN_8BIT", "true")).lower() in ("1", "true", "yes")
        )
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", temperature))
        self.top_p = float(os.getenv("MODEL_TOP_P", top_p))
        self.max_new_tokens = int(os.getenv("MODEL_MAX_NEW_TOKENS", max_new_tokens))

        if self.device == "auto" and not torch.cuda.is_available():
            self.device = "cpu"

        if self.device == "cpu":
            self.load_in_8bit = False

        self.tokenizer = None
        self.model = None
        self.generator = None

        self._prepare_model()

    def _model_artifacts_exist(self) -> bool:
        if not self.model_local_path.exists():
            return False

        if (self.model_local_path / "config.json").exists():
            return True
        if any(self.model_local_path.glob("*.bin")):
            return True
        if any(self.model_local_path.glob("*.safetensors")):
            return True
        return False

    def _prepare_model(self):
        if self._model_artifacts_exist():
            logger.info("Loading model from local path: %s", self.model_local_path)
            model_source = str(self.model_local_path)
        elif self.s3_bucket and self.s3_prefix:
            logger.info(
                "Local model not found. Downloading model from S3 bucket %s prefix %s",
                self.s3_bucket,
                self.s3_prefix,
            )
            self._download_model_from_s3()
            model_source = str(self.model_local_path)
        elif self.hf_model_id:
            if self.hf_token:
                hf_login(token=self.hf_token)
            logger.info("Downloading model from Hugging Face: %s", self.hf_model_id)
            self.model_local_path.mkdir(parents=True, exist_ok=True)
            model_source = self.hf_model_id
        else:
            raise ValueError(
                "No model source configured. Set MODEL_LOCAL_PATH, or S3_MODEL_BUCKET and S3_MODEL_PREFIX, or HUGGINGFACE_MODEL_ID."
            )

        self.model_local_path.mkdir(parents=True, exist_ok=True)

        logger.info("Loading tokenizer from %s", model_source)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(self.model_local_path),
            use_fast=True,
            trust_remote_code=False,
        )

        model_kwargs = {
            "cache_dir": str(self.model_local_path),
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }

        if self.device in ("auto", "cuda"):
            model_kwargs["device_map"] = "auto"
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            else:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["device_map"] = None
            model_kwargs["torch_dtype"] = torch.float32

        logger.info("Loading model from %s with device %s", model_source, self.device)
        self.model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

        pipeline_device = -1
        if self.device == "cuda":
            pipeline_device = 0
        elif self.device == "auto":
            pipeline_device = "auto"

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
            trust_remote_code=False,
        )

    def _download_model_from_s3(self):
        if not self.s3_bucket or not self.s3_prefix:
            raise ValueError("S3_MODEL_BUCKET and S3_MODEL_PREFIX must be set to download from S3.")

        client = boto3.client("s3")
        prefix = self.s3_prefix.rstrip("/") + "/"
        self.model_local_path.mkdir(parents=True, exist_ok=True)

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                relative_path = key[len(prefix) :].lstrip("/")
                destination = self.model_local_path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading s3://%s/%s to %s", self.s3_bucket, key, destination)
                client.download_file(self.s3_bucket, key, str(destination))

    def generate_response(self, prompt: str, max_length: int = 1024) -> Dict[str, Any]:
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False,
                return_full_text=False,
            )

            if isinstance(outputs, list) and len(outputs) > 0:
                text = outputs[0].get("generated_text") or outputs[0].get("text") or ""
            else:
                text = str(outputs)

            return {"response": text.strip(), "status": "success"}
        except Exception as e:
            logger.error("Error generating response from local model: %s", e)
            return {"response": None, "status": "error", "message": str(e)}
