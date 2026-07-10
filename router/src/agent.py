"""Core routing agent with Qwen API client and Dynamic Procurement"""

import os
import re
import logging
from typing import Dict, Any, Tuple

from .qwen_client import QwenClient
from .clients.fireworks import FireworksClient
from .qwen_router import ModelRegistry

logger = logging.getLogger(__name__)


class RoutingAgent:
    def __init__(self):
        self.qwen = QwenClient()
        self.fireworks = FireworksClient(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url=os.environ.get("FIREWORKS_BASE_URL"),
        )
        self.metrics = {
            "total_queries": 0,
            "local_queries": 0,
            "fireworks_queries": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

        # Initialize dynamic model procurement
        self._initialize_models()

    def _initialize_models(self):
        """Load ALLOWED_MODELS, cross-reference registry, and sort by capability"""
        registry = ModelRegistry()
        env_models = os.environ.get("ALLOWED_MODELS", "glm-5p1,deepseek-v4-pro")
        model_ids = [m.strip() for m in env_models.split(",") if m.strip()]

        self.available_models = []
        for m_id in model_ids:
            model = registry.get_model(m_id)
            if model:
                self.available_models.append(model)
            else:
                # Safe fallback if a model is in .env but missing from our registry
                logger.warning(
                    f"Model {m_id} missing from registry. Assigning default fallback score."
                )
                self.available_models.append(
                    {
                        "id": m_id,
                        "display_name": m_id.upper(),
                        "cost_per_1k_input": 0.001,
                        "cost_per_1k_output": 0.001,
                        "capability_score": 0.8,  # Assume medium-high capability
                    }
                )

        # The Magic: Sort from weakest/cheapest to strongest/most expensive
        self.available_models.sort(key=lambda x: x.get("capability_score", 0.0))
        logger.info(
            f"Loaded {len(self.available_models)} allowed models, sorted by capability."
        )

    def _select_model(self, score: float) -> Dict:
        """Map Qwen's complexity score (0.3 to 1.0) to the sorted model array"""
        if not self.available_models:
            raise ValueError("No models available for routing")

        # Normalize the score window (0.3 -> 1.0 becomes 0.0 -> 1.0)
        normalized_score = max(0.0, (score - 0.3) / 0.7)

        # Calculate array index based on score percentage
        index = int(normalized_score * len(self.available_models))
        # Ensure we don't go out of bounds if score is exactly 1.0
        index = min(len(self.available_models) - 1, index)

        return self.available_models[index]

    def _call_fireworks(self, prompt: str, model: Dict) -> Tuple[str, int, float]:
        """Execute the prompt on the dynamically selected Fireworks model"""
        try:
            response = self.fireworks.generate(model, prompt, max_tokens=2048)

            from .metrics.counter import calculate_cost

            cost = calculate_cost(
                response.input_tokens,
                response.output_tokens,
                model["cost_per_1k_input"],
                model["cost_per_1k_output"],
            )

            # Accumulate metrics for the dashboard
            self.metrics["total_tokens"] += response.total_tokens
            self.metrics["total_cost"] += cost

            return response.text, response.total_tokens, cost

        except Exception as e:
            logger.error(f"Fireworks call failed: {e}")
            return f"Error: {str(e)}", 0, 0.0

    def process_query(self, prompt: str) -> Dict[str, Any]:
        self.metrics["total_queries"] += 1

        try:
            # 1. Gatekeeper Classification via Qwen API
            classification = self.qwen.classify(prompt)
            raw_text = classification.get("text", "0.5").strip()

            # X-RAY Regex: Surgically extract only the numbers, ignoring words
            numbers = re.findall(r"[\d.]+", raw_text)
            score = float(numbers[0]) if numbers else 0.5

            # Clamp the score strictly between 0.0 and 1.0
            score = min(1.0, max(0.0, score))

            logger.info(
                f"🔍 X-RAY: Raw Qwen Output: '{raw_text}' -> Parsed Score: {score:.2f}"
            )

            # 2. Autonomous Routing Logic
            if score < 0.3:
                # Qwen knows it can solve this easily -> Local Execution (Zero Token Cost)
                response = self.qwen.generate(prompt)
                model_used = "Qwen (local)"
                tokens = 0
                cost = 0.0
                self.metrics["local_queries"] += 1
                logger.info(
                    f"📌 ROUTE: local - Zero cost execution (score: {score:.2f})"
                )

            else:
                # Complex Prompt -> Procure Model -> Compress -> External Execution
                selected_model = self._select_model(score)
                model_id = selected_model["id"]
                logger.info(
                    f"📌 ROUTE: fireworks - Score {score:.2f} dynamically mapped to {model_id}"
                )

                # Universal Compression to slash token costs
                compressed_prompt = self.qwen.compress(prompt)

                # Execute on external API
                response, tokens, cost = self._call_fireworks(
                    compressed_prompt, selected_model
                )
                model_used = selected_model.get("display_name", model_id)
                self.metrics["fireworks_queries"] += 1

            return {
                "response": response,
                "model": model_used,
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
