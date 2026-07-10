# Unified Container - Qwen Server + Router
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install all dependencies
RUN pip install --no-cache-dir --prefer-binary \
    fastapi uvicorn httpx \
    torch transformers accelerate bitsandbytes optimum sentencepiece protobuf \
    python-dotenv \
    tenacity tiktoken pydantic pyyaml \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download Qwen model during build (simple version)
RUN python -c "\
import os; \
os.environ['HF_HOME'] = '/app/.cache/huggingface'; \
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'; \
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig; \
print('=' * 60); \
print('📥 DOWNLOADING QWEN2.5-1.5B MODEL'); \
print('=' * 60); \
print('⏳ This will take 3-5 minutes...'); \
print('📦 Model size: ~1.5 GB'); \
print('=' * 60); \
print(''); \
print('🔄 Loading tokenizer...'); \
tokenizer = AutoTokenizer.from_pretrained( \
    'Qwen/Qwen2.5-1.5B-Instruct', \
    trust_remote_code=True, \
    cache_dir='/app/.cache/huggingface' \
); \
print('✅ Tokenizer loaded!'); \
print(''); \
print('🔄 Loading model (4-bit quantization)...'); \
bnb_config = BitsAndBytesConfig( \
    load_in_4bit=True, \
    bnb_4bit_quant_type='nf4', \
    bnb_4bit_compute_dtype='float16', \
    bnb_4bit_use_double_quant=True \
); \
model = AutoModelForCausalLM.from_pretrained( \
    'Qwen/Qwen2.5-1.5B-Instruct', \
    quantization_config=bnb_config, \
    device_map='auto', \
    torch_dtype='float16', \
    trust_remote_code=True, \
    cache_dir='/app/.cache/huggingface' \
); \
print('✅ Model loaded!'); \
print(''); \
print('=' * 60); \
print('✅ QWEN MODEL PRE-DOWNLOADED SUCCESSFULLY!'); \
print('=' * 60); \
print(f'📊 Model parameters: {model.num_parameters():,}'); \
print(f'📁 Cache location: /app/.cache/huggingface'); \
print('=' * 60); \
"

# Copy source code from router/src to /app/src
COPY router/src/ ./src/

# Create cache directory and set permissions
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app /app/.cache
USER agent

# Entrypoint
ENTRYPOINT ["python", "-m", "src.main"]
CMD []