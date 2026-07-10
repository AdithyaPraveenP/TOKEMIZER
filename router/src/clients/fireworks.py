"""Fireworks AI API Client"""

import os
import httpx
import logging
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class FireworksResponse:
    """Fireworks API response wrapper"""

    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        self.input_tokens = usage.get("prompt_tokens", 0)
        self.output_tokens = usage.get("completion_tokens", 0)
        self.total_tokens = usage.get(
            "total_tokens", self.input_tokens + self.output_tokens
        )


class FireworksClient:
    """Fireworks AI API Client with retry logic"""

    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self, model: Dict, prompt: str, max_tokens: int = 4096
    ) -> FireworksResponse:
        """Generate completion from Fireworks"""
        model_path = f"accounts/fireworks/models/{model['id']}"

        payload = {
            "model": model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_tokens, model.get("max_tokens", 4096)),
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        url = f"{self.base_url}/chat/completions"

        logger.debug(f"Calling {model['display_name']} with {len(prompt)} chars")

        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        return FireworksResponse(response.json())

    def close(self):
        self.client.close()
