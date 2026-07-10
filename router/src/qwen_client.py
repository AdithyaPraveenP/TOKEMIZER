import os
import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8000")


class QwenClient:
    """Client for Qwen inference server"""

    def __init__(self, base_url: str = QWEN_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=600)

    def classify(self, query: str) -> Dict[str, Any]:
        """Classify query complexity"""
        prompt = f"""Rate complexity from 0-1. 0=simple, 1=complex.
Query: {query}
Score:"""
        try:
            response = self.client.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
            )
            data = response.json()
            # Return with 'text' field
            return {"text": data.get("text", "0.5")}
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {"text": "0.5"}

    def generate(self, query: str, max_tokens: int = 2048) -> str:
        """Generate response"""
        try:
            response = self.client.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": query,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )
            return response.json().get("text", "Error: No response")
        except Exception as e:
            logger.error(f"Generate failed: {e}")
            return f"Error: {str(e)}"

    def compress(self, query: str) -> str:
        """Compress query"""
        prompt = f"""Compress this to 50% length. Preserve key info.
Original: {query}
Compressed:"""
        try:
            response = self.client.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max(64, len(query.split())),
                    "temperature": 0.3,
                },
            )
            return response.json().get("text", query)
        except Exception as e:
            logger.error(f"Compress failed: {e}")
            return query

    def refine(self, query: str) -> str:
        """Refine complex query"""
        prompt = f"""Refine this query clearly:
Original: {query}
Refined:"""
        try:
            response = self.client.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.2,
                },
            )
            return response.json().get("text", query)
        except Exception as e:
            logger.error(f"Refine failed: {e}")
            return query

    def close(self):
        self.client.close()
