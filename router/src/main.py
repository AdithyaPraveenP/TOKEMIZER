#!/usr/bin/env python3
"""Unified Entrypoint - Qwen pre-downloaded, starts fast with Data Hot-Reload"""

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
    """Wait for Qwen server to be ready (10 minutes max)"""
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


def process_batch(agent: RoutingAgent):
    """Process a single batch of tasks"""
    tasks = load_tasks()
    logger.info(f"📝 Loaded {len(tasks)} tasks for processing")

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
    logger.info("📊 Batch Summary:")
    logger.info(f"   Queries processed: {metrics.get('total_queries', 0)}")
    logger.info(f"   Local queries: {metrics.get('local_queries', 0)} (0 tokens!)")
    logger.info(f"   Fireworks queries: {metrics.get('fireworks_queries', 0)}")
    logger.info(f"   Total tokens: {metrics.get('total_tokens', 0)}")
    logger.info(f"   Total cost: ${metrics.get('total_cost', 0.0):.6f}")
    logger.info("=" * 60)


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

    try:
        # Step 1: Start Qwen server
        qwen_process = start_qwen_server()

        # Step 2: Wait for Qwen to be ready
        if not wait_for_qwen_server(timeout_seconds=600):
            logger.error("❌ Qwen server failed to start. Exiting.")
            qwen_process.terminate()
            sys.exit(1)

        # Step 3: Initialize Agent
        agent = RoutingAgent()
        logger.info("✅ Agent ready. Watching for file changes...")

        # Step 4: Infinite Data-Watcher Loop
        last_mtime = 0
        while True:
            try:
                # Check if tasks.json exists and get its last modified time
                if INPUT_PATH.exists():
                    current_mtime = os.path.getmtime(INPUT_PATH)
                    if current_mtime > last_mtime:
                        logger.info("🔄 Detected change in tasks.json. Processing...")
                        process_batch(agent)
                        last_mtime = current_mtime

                time.sleep(2)  # Sleep for 2 seconds before checking again

            except KeyboardInterrupt:
                logger.info("🛑 Shutting down gracefully...")
                break
            except Exception as loop_err:
                logger.error(f"Error in processing loop: {loop_err}")
                time.sleep(5)  # Back off slightly on error

    except SystemExit:
        # Catch intentional exits (like missing env vars) so they don't get swallowed
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if "agent" in locals() and hasattr(agent, "close"):
            agent.close()
        if "qwen_process" in locals():
            qwen_process.terminate()


if __name__ == "__main__":
    print("🔧 X-RAY: main.py is successfully executing!")
    main()
