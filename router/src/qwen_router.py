"""Qwen-Only Routing Agent - Optimized for 4GB RAM"""

import os
import re
import logging
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .clients.fireworks import FireworksClient
from .metrics.counter import count_tokens, calculate_cost

logger = logging.getLogger(__name__)

# Model cache directory (for Docker)
MODEL_CACHE = os.environ.get("HF_HOME", "/app/.cache/huggingface")


class QwenRouter:
    """
    Single model router using Qwen2.5-1.5B
    Fits in 4GB RAM with 4-bit quantization
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load Qwen model
        self._load_model()

        # Fireworks client (for complex queries)
        self.fireworks = FireworksClient(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url=os.environ.get("FIREWORKS_BASE_URL"),
        )

        # Metrics
        self.metrics = {
            "local_queries": 0,
            "fireworks_queries": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "compressed_tokens_saved": 0,
        }

    def _load_model(self):
        """Load Qwen2.5-1.5B with 4-bit quantization"""
        logger.info("Loading Qwen2.5-1.5B (4-bit quantization)...")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                trust_remote_code=True,
                cache_dir=MODEL_CACHE,
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("✅ Qwen2.5-1.5B loaded successfully")

            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory used: {allocated:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query complexity using Qwen (zero tokens!)

        Returns:
            dict with score (0-1), complexity, confidence, required_capability
        """
        # For very short queries, fast-path classification
        if len(query.split()) < 5:
            return {
                "score": 0.1,
                "complexity": "simple",
                "confidence": 0.95,
                "required_capability": 0.60,
                "reasoning": "Very short query - likely simple",
            }

        prompt = f"""You are a classifier. Rate this query's complexity from 0 to 1.

0 = simple factoid question (who, what, when, where, define, list, convert)
1 = complex multi-step reasoning (analyze, design, derive, implement, debug, optimize)

Query: {query}

Respond ONLY with a number between 0 and 1. No explanation."""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract number
            numbers = re.findall(r"[\d.]+", response)
            score = float(numbers[0]) if numbers else 0.5
            score = min(1.0, max(0.0, score))

        except Exception as e:
            logger.warning(f"Classification failed: {e}, using heuristic fallback")
            score = self._heuristic_classify(query)

        # Determine complexity level
        if score < 0.25:
            complexity = "simple"
            confidence = 0.92
            required_capability = 0.60
            reasoning = "Low complexity query"
        elif score < 0.55:
            complexity = "medium"
            confidence = 0.85
            required_capability = 0.75
            reasoning = "Medium complexity query"
        else:
            complexity = "complex"
            confidence = 0.78
            required_capability = 0.85
            reasoning = "High complexity query"

        logger.info(f"Classification: {complexity} (score: {score:.2f})")

        return {
            "score": score,
            "complexity": complexity,
            "confidence": confidence,
            "required_capability": required_capability,
            "reasoning": reasoning,
        }

    def _heuristic_classify(self, query: str) -> float:
        """Fallback heuristic classification"""
        text = query.lower()

        # Simple indicators
        simple_keywords = ["what", "who", "when", "where", "define", "list", "convert"]
        complex_keywords = [
            "analyze",
            "design",
            "derive",
            "implement",
            "debug",
            "optimize",
        ]

        simple_score = sum(1 for kw in simple_keywords if kw in text) / len(
            simple_keywords
        )
        complex_score = sum(1 for kw in complex_keywords if kw in text) / len(
            complex_keywords
        )

        # Length heuristic
        word_count = len(text.split())
        if word_count < 10:
            simple_score += 0.3
        elif word_count > 50:
            complex_score += 0.3

        total = simple_score + complex_score or 1
        return complex_score / total

    def execute(self, query: str, max_tokens: int = 2048) -> str:
        """Execute simple query with Qwen (zero tokens!)"""
        messages = [{"role": "user", "content": query}]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 2048),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            response = response[len(text) :].strip()

            self.metrics["local_queries"] += 1
            logger.info(f"✅ Qwen local execution: {len(response)} chars")

            return response if response else "I couldn't generate a response."

        except Exception as e:
            logger.error(f"Qwen execution failed: {e}")
            return f"Error: {str(e)}"

    def compress(self, query: str) -> str:
        """Compress query for Fireworks (reduce input tokens)"""
        prompt = f"""Compress this query to ~50% of its length. Preserve ALL technical terms, code, numbers, and key context.
Remove filler words, pleasantries, and redundancy.

Original: {query}

Compressed:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max(64, len(query.split())),
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the compressed part
            compressed = response.split("Compressed:")[-1].strip()

            # Calculate tokens saved
            original_tokens = count_tokens(query)
            compressed_tokens = count_tokens(compressed)
            saved = original_tokens - compressed_tokens
            self.metrics["compressed_tokens_saved"] += max(0, saved)

            logger.info(
                f"Compressed: {original_tokens} → {compressed_tokens} tokens (saved {saved})"
            )

            return compressed if compressed and len(compressed) > 10 else query

        except Exception as e:
            logger.warning(f"Compression failed: {e}, using original")
            return query

    def refine(self, query: str) -> str:
        """Refine complex query for top Fireworks model"""
        prompt = f"""Refine this query to be crystal clear. Extract:
1. The core requirement
2. Any constraints or limitations
3. Expected output format
4. Key domain terms

Original: {query}

Refined query:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            refined = response.split("Refined query:")[-1].strip()

            return refined if refined and len(refined) > 10 else query

        except Exception as e:
            logger.warning(f"Refinement failed: {e}, using original")
            return query

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Full query processing pipeline with intelligent routing

        Returns:
            dict with response, model, tokens, cost, classification
        """
        # 1. Classify (zero tokens)
        classification = self.classify(query)
        score = classification["score"]

        # 2. Route based on score
        if score < 0.3:
            # Simple → Qwen executes locally (FREE!)
            response = self.execute(query)
            tokens_used = 0
            cost = 0.0
            model = "Qwen2.5-1.5B (local)"
            route = "local"

            self.metrics["local_queries"] += 1
            logger.info(f"📌 ROUTE: {route} - Simple query, local execution")

        elif score < 0.6:
            # Medium → Compress then Fireworks
            compressed = self.compress(query)
            # Use cheap Fireworks model
            response, tokens, cost = self._call_fireworks(compressed, "cheap")
            model = response.get("model", "Fireworks (cheap)")
            route = "fireworks_cheap"

            self.metrics["fireworks_queries"] += 1
            self.metrics["total_tokens"] += tokens
            self.metrics["total_cost"] += cost
            logger.info(
                f"📌 ROUTE: {route} - Medium query, compressed {len(query)}→{len(compressed)} chars"
            )

        else:
            # Complex → Refine then Top Fireworks
            refined = self.refine(query)
            response, tokens, cost = self._call_fireworks(refined, "top")
            model = response.get("model", "Fireworks (top)")
            route = "fireworks_top"

            self.metrics["fireworks_queries"] += 1
            self.metrics["total_tokens"] += tokens
            self.metrics["total_cost"] += cost
            logger.info(f"📌 ROUTE: {route} - Complex query, refined for clarity")

        return {
            "response": (
                response.get("text", str(response))
                if isinstance(response, dict)
                else str(response)
            ),
            "model": model,
            "tokens": tokens_used if route == "local" else tokens,
            "cost": cost,
            "route": route,
            "classification": classification,
            "tokens_saved": self.metrics["compressed_tokens_saved"],
        }

    def _call_fireworks(self, prompt: str, tier: str) -> tuple:
        """Call Fireworks API with appropriate model"""
        try:
            # Select model based on tier
            if tier == "cheap":
                # Use cheapest allowed model
                model_id = "glm-5p1"  # Cheapest
            else:
                # Use most capable allowed model
                model_id = "deepseek-v4-pro"  # Most capable

            # Get model profile
            from .router.registry import ModelRegistry

            registry = ModelRegistry()
            model = registry.get_model(model_id)

            if not model:
                raise ValueError(f"Model {model_id} not found")

            # Call Fireworks
            response = self.fireworks.generate(model, prompt, max_tokens=2048)

            # Calculate cost
            cost = calculate_cost(
                response.input_tokens,
                response.output_tokens,
                model["cost_per_1k_input"],
                model["cost_per_1k_output"],
            )

            self.metrics["total_tokens"] += response.total_tokens
            self.metrics["total_cost"] += cost

            return (
                {
                    "text": response.text,
                    "model": model["display_name"],
                    "tokens": response.total_tokens,
                },
                response.total_tokens,
                cost,
            )

        except Exception as e:
            logger.error(f"Fireworks call failed: {e}")
            return {"text": f"Error: {str(e)}", "model": "error"}, 0, 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get session metrics"""
        return self.metrics

    def close(self):
        """Clean up resources"""
        if hasattr(self, "fireworks"):
            self.fireworks.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelRegistry:
    """Simple model registry for Fireworks models"""

    DEFAULT_MODELS = {
        "glm-5p1": {
            "id": "glm-5p1",
            "display_name": "GLM 5.1",
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.00020,
            "capability_score": 0.75,
        },
        "deepseek-v4-pro": {
            "id": "deepseek-v4-pro",
            "display_name": "DeepSeek V4 Pro",
            "cost_per_1k_input": 0.00100,
            "cost_per_1k_output": 0.00120,
            "capability_score": 0.92,
        },
    }

    def __init__(self):
        self.models = self.DEFAULT_MODELS

    def get_model(self, model_id: str) -> Optional[Dict]:
        return self.models.get(model_id)
