"""Core routing agent with Qwen API client"""

import os
import logging
from typing import Dict, Any, Tuple

from .qwen_client import QwenClient
from .clients.fireworks import FireworksClient

logger = logging.getLogger(__name__)


class RoutingAgent:
    def __init__(self):
        self.qwen = QwenClient()
        self.fireworks = FireworksClient(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url=os.environ.get("FIREWORKS_BASE_URL"),
        )
        self.metrics = {"total_queries": 0, "local_queries": 0, "fireworks_queries": 0}

    def _call_fireworks(self, prompt: str, tier: str) -> Tuple[str, int, float]:
        """Call Fireworks API with appropriate model"""
        try:
            # Select model based on tier
            if tier == "cheap":
                model_id = "glm-5p1"  # Cheapest allowed model
            else:
                model_id = "deepseek-v4-pro"  # Most capable

            # Get model profile from registry
            from .qwen_router import ModelRegistry

            registry = ModelRegistry()
            model = registry.get_model(model_id)

            if not model:
                logger.error(f"Model {model_id} not found")
                return f"Error: Model {model_id} not found", 0, 0.0

            # Call Fireworks
            response = self.fireworks.generate(model, prompt, max_tokens=2048)

            # Calculate cost
            from .metrics.counter import calculate_cost

            cost = calculate_cost(
                response.input_tokens,
                response.output_tokens,
                model["cost_per_1k_input"],
                model["cost_per_1k_output"],
            )

            return response.text, response.total_tokens, cost

        except Exception as e:
            logger.error(f"Fireworks call failed: {e}")
            return f"Error: {str(e)}", 0, 0.0

    def process_query(self, prompt: str) -> Dict[str, Any]:
        self.metrics["total_queries"] += 1

        try:
            # 1. Classify via Qwen API
            classification = self.qwen.classify(prompt)
            score = float(classification.get("text", "0.5").strip() or "0.5")
            logger.info(f"Classification score: {score:.2f}")

            # 2. Route based on score
            if score < 0.3:
                # Simple → Qwen local (0 tokens!)
                response = self.qwen.generate(prompt)
                model = "Qwen (local)"
                tokens = 0
                cost = 0.0
                self.metrics["local_queries"] += 1
                logger.info(f"📌 ROUTE: local - Simple query (score: {score:.2f})")

            elif score < 0.6:
                # Medium → Compress → Fireworks cheap
                compressed = self.qwen.compress(prompt)
                response, tokens, cost = self._call_fireworks(compressed, "cheap")
                model = "Fireworks (cheap)"
                self.metrics["fireworks_queries"] += 1
                logger.info(
                    f"📌 ROUTE: fireworks_cheap - Medium query (score: {score:.2f})"
                )

            else:
                # Complex → Refine → Fireworks top
                refined = self.qwen.refine(prompt)
                response, tokens, cost = self._call_fireworks(refined, "top")
                model = "Fireworks (top)"
                self.metrics["fireworks_queries"] += 1
                logger.info(
                    f"📌 ROUTE: fireworks_top - Complex query (score: {score:.2f})"
                )

            return {
                "response": response,
                "model": model,
                "tokens": tokens,
                "cost": cost,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "model": "error",
                "tokens": 0,
                "cost": 0.0,
                "success": False,
            }

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def close(self):
        self.qwen.close()
        self.fireworks.close()
