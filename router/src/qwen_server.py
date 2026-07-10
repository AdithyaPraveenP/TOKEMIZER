#!/usr/bin/env python3
"""Qwen Inference Server - Model pre-downloaded during build"""

import os
import logging
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Inference Server")

MODEL_CACHE = os.environ.get("HF_HOME", "/app/.cache/huggingface")

logger.info("=" * 60)
logger.info("🧠 LOADING QWEN2.5-1.5B FROM CACHE")
logger.info("=" * 60)

try:
    start_time = time.time()

    logger.info("📂 Checking cache directory...")
    if os.path.exists(MODEL_CACHE):
        logger.info(f"✅ Cache found at: {MODEL_CACHE}")
    else:
        logger.warning(f"⚠️ Cache not found at: {MODEL_CACHE}")

    logger.info("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✅ Tokenizer loaded!")

    logger.info("🔄 Loading model (4-bit quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
    )

    load_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("✅ QWEN LOADED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"📊 Model parameters: {model.num_parameters():,}")
    logger.info(f"⏱️  Load time: {load_time:.2f} seconds")
    logger.info(f"📁 Cache location: {MODEL_CACHE}")
    logger.info("=" * 60)

except Exception as e:
    logger.error("=" * 60)
    logger.error("❌ FAILED TO LOAD QWEN FROM CACHE")
    logger.error("=" * 60)
    logger.error(f"Error: {e}")
    logger.error("")
    logger.error("Possible causes:")
    logger.error("  1. Model not pre-downloaded during Docker build")
    logger.error("  2. Cache directory permissions issue")
    logger.error("  3. Corrupted model files")
    logger.error("")
    logger.error("Fix: Rebuild with --no-cache")
    raise


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    text: str
    tokens: int


@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    try:
        messages = [{"role": "user", "content": request.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = request.temperature > 0.0
        temperature = request.temperature if do_sample else 1.0

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return InferenceResponse(text=response, tokens=len(response) // 4)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "Qwen2.5-1.5B"}


@app.get("/")
async def root():
    return {"message": "Qwen Inference Server is running", "model": "Qwen2.5-1.5B"}


def run_server():
    """Run the server"""
    logger.info("🚀 Starting Uvicorn server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    run_server()
