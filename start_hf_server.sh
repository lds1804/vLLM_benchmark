#!/bin/bash

# Target model for your benchmark: mistralai/Mistral-7B-Instruct-v0.1
# The port is set to 8000 (the default used by your python script)

echo "Installing dependencies for native Hugging Face server (FastAPI, Uvicorn, Accelerate)..."
pip install fastapi uvicorn accelerate

echo "Initializing the native Hugging Face Transformers server..."
echo "This will download the weights if not cached and run via standard Transformers (without vLLM optimizations)."

python hf_api_server.py
