"""Token counter using tiktoken"""

import tiktoken
import logging

logger = logging.getLogger(__name__)

try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None
    logger.warning("Tiktoken not available, using character-based fallback")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or fallback"""
    if not text:
        return 0
    if ENCODER:
        return len(ENCODER.encode(text))
    # Fallback: ~4 chars per token
    return len(text) // 4


def count_input_tokens(prompt: str) -> int:
    """Count input tokens"""
    return count_tokens(prompt)


def count_output_tokens(response: str) -> int:
    """Count output tokens"""
    return count_tokens(response)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
) -> float:
    """Calculate cost in USD"""
    return (input_tokens / 1000 * cost_per_1k_input) + (
        output_tokens / 1000 * cost_per_1k_output
    )


def format_cost(cost: float) -> str:
    """Format cost as readable string"""
    if cost < 0.00001:
        return "$0.00000"
    if cost < 0.001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.5f}"
    return f"${cost:.4f}"
