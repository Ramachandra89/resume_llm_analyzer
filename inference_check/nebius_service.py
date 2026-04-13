"""
Nebius AI inference service.
Uses the OpenAI-compatible API provided by Nebius AI Studio.

Supported models (as of 2025):
  - meta-llama/Meta-Llama-3.1-70B-Instruct
  - meta-llama/Meta-Llama-3.1-8B-Instruct
  - Qwen/Qwen2.5-72B-Instruct
  - mistralai/Mixtral-8x7B-Instruct-v0.1
  - deepseek-ai/DeepSeek-R1

Set NEBIUS_API_KEY in your environment or pass it directly.
"""

import os
import re
from typing import Optional

from openai import OpenAI

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Reasoning models emit <think>...</think> blocks before the actual answer.
# Strip them so callers only see the final response.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# These models need extra token headroom for internal reasoning chains.
_REASONING_MODELS = {"nvidia/nemotron-3-super-120b-a12b"}
_REASONING_EXTRA_TOKENS = 4096


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class NebiusService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.6,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY")
        if not self.api_key:
            raise ValueError("NEBIUS_API_KEY is required")

        self.model = model
        self.temperature = temperature
        # Reasoning models need extra budget for their internal chain-of-thought.
        if model in _REASONING_MODELS:
            self.max_tokens = max_tokens + _REASONING_EXTRA_TOKENS
        else:
            self.max_tokens = max_tokens

        self.client = OpenAI(
            base_url=NEBIUS_BASE_URL,
            api_key=self.api_key,
        )

    def _effective_max_tokens(self, max_tokens: Optional[int]) -> int:
        base = max_tokens if max_tokens is not None else self.max_tokens
        if max_tokens is not None and self.model in _REASONING_MODELS:
            return base + _REASONING_EXTRA_TOKENS
        return base

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> dict:
        """
        Send a prompt to Nebius and return the response text.

        Returns:
            {"response": str, "status": "success" | "error", "error": str | None}
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self._effective_max_tokens(max_tokens),
            )
            text = completion.choices[0].message.content or ""
            finish_reason = completion.choices[0].finish_reason
            response = _strip_thinking(text)
            result = {"response": response, "status": "success"}
            if finish_reason == "length":
                result["truncated"] = True
            return result
        except Exception as exc:
            return {"response": "", "status": "error", "error": str(exc)}

    def chat(self, messages: list[dict], max_tokens: Optional[int] = None) -> dict:
        """
        Multi-turn chat with Nebius.

        Args:
            messages: list of {"role": "user"|"assistant"|"system", "content": str}
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self._effective_max_tokens(max_tokens),
            )
            text = completion.choices[0].message.content or ""
            finish_reason = completion.choices[0].finish_reason
            response = _strip_thinking(text)
            result = {"response": response, "status": "success"}
            if finish_reason == "length":
                result["truncated"] = True
            return result
        except Exception as exc:
            return {"response": "", "status": "error", "error": str(exc)}

    @staticmethod
    def available_models() -> list[str]:
        return [
            "meta-llama/Llama-3.3-70B-Instruct",
            "nvidia/nemotron-3-super-120b-a12b",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
        ]
