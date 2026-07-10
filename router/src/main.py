#!/usr/bin/env python3
"""Unified Entrypoint - Qwen pre-downloaded, starts fast"""

import json
import os
import sys
import time
import logging
import subprocess
import threading
from pathlib import Path
import httpx

from dotenv import load_dotenv

load_dotenv()

from .agent import RoutingAgent

INPUT_PATH = Path("/input/tasks.json")
OUTPUT_PATH = Path("/output/results.json")
QWEN_HEALTH_URL = "http://127.0.0.1:8000/health"
QWEN_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "qwen_server.py")

PRACTICE_TASKS = [
    {
        "task_id": "practice-01",
        "prompt": "What is the capital of Australia, and what body of water is it near?",
    },
    {
        "task_id": "practice-02",
        "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?",
    },
    {
        "task_id": "practice-03",
        "prompt": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.",
    },
    {
        "task_id": "practice-04",
        "prompt": "Summarize the following in exactly one sentence: Artificial intelligence is transforming how businesses operate.",
    },
    {
        "task_id": "practice-05",
        "prompt": "Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.",
    },
    {
        "task_id": "practice-06",
        "prompt": "This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.",
    },
    {
        "task_id": "practice-07",
        "prompt": "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?",
    },
    {
        "task_id": "practice-08",
        "prompt": "Write a Python function that returns the second-largest number in a list, handling duplicates correctly.",
    },
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def start_qwen_server():
    """Start Qwen server as subprocess"""
    logger.info("🚀 Starting Qwen inference server...")

    process = subprocess.Popen(
        [sys.executable, QWEN_SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def log_output():
        for line in process.stdout:
            logger.info(f"[QWEN] {line.strip()}")

    threading.Thread(target=log_output, daemon=True).start()
    return process


def wait_for_qwen_server(timeout_seconds=600):
    """Wait for Qwen server to be ready (2 minutes max)"""
    logger.info(f"⏳ Waiting for Qwen server (timeout: {timeout_seconds}s)...")

    start_time = time.time()
    client = httpx.Client(timeout=5)

    while time.time() - start_time < timeout_seconds:
        try:
            response = client.get(QWEN_HEALTH_URL)
            if response.status_code == 200:
                logger.info("✅ Qwen server is ready!")
                client.close()
                return True
        except Exception:
            pass

        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 and elapsed > 0:
            logger.info(f"⏳ Waiting for Qwen server... ({elapsed}s elapsed)")
        time.sleep(2)

    client.close()
    logger.error(f"❌ Qwen server failed to start within {timeout_seconds}s")
    return False


def load_tasks() -> list:
    if INPUT_PATH.exists():
        with open(INPUT_PATH, "r") as f:
            return json.load(f)
    logger.info("No input file found, using practice tasks")
    return PRACTICE_TASKS


def save_results(results: list) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Saved {len(results)} results to {OUTPUT_PATH}")


def main():
    logger.info("=" * 60)
    logger.info("🚀 Hybrid Token Router - AMD Hackathon ACT II")
    logger.info("📦 Qwen2.5-1.5B (4-bit) - Pre-downloaded in image")
    logger.info("=" * 60)

    required = ["FIREWORKS_API_KEY", "FIREWORKS_BASE_URL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        logger.error(f"Missing required env vars: {missing}")
        sys.exit(1)

    logger.info(f"Base URL: {os.environ.get('FIREWORKS_BASE_URL')}")
    logger.info(f"Allowed Models: {os.environ.get('ALLOWED_MODELS', 'default')}")

    try:
        # Step 1: Start Qwen server (model already in image)
        qwen_process = start_qwen_server()

        # Step 2: Wait for Qwen to be ready (2 min max)
        if not wait_for_qwen_server(timeout_seconds=600):
            logger.error("❌ Qwen server failed to start. Exiting.")
            qwen_process.terminate()
            sys.exit(1)

        # Step 3: Process tasks
        tasks = load_tasks()
        logger.info(f"📝 Loaded {len(tasks)} tasks")

        agent = RoutingAgent()
        logger.info("✅ Agent ready")

        results = []
        for i, task in enumerate(tasks, 1):
            task_id = task.get("task_id", f"task-{i}")
            prompt = task.get("prompt", "")

            logger.info(f"\n📌 [{i}/{len(tasks)}] {task_id}")

            if not prompt:
                results.append({"task_id": task_id, "answer": "Error: Empty prompt"})
                continue

            result = agent.process_query(prompt)
            results.append(
                {
                    "task_id": task_id,
                    "answer": result.get("response", "Error: No response"),
                }
            )

        save_results(results)

        metrics = agent.get_metrics()
        logger.info("\n" + "=" * 60)
        logger.info("📊 Summary:")
        logger.info(f"   Queries processed: {metrics.get('total_queries', 0)}")
        logger.info(f"   Local queries: {metrics.get('local_queries', 0)} (0 tokens!)")
        logger.info(f"   Fireworks queries: {metrics.get('fireworks_queries', 0)}")
        logger.info(f"   Total tokens: {metrics.get('total_tokens', 0)}")
        logger.info(f"   Total cost: ${metrics.get('total_cost', 0.0):.6f}")
        logger.info("=" * 60)
        logger.info("✅ Done! Exiting with code 0")

        agent.close()
        qwen_process.terminate()
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
