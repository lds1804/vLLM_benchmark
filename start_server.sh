#!/bin/bash

# Target model for your benchmark: mistralai/Mistral-7B-Instruct-v0.1
# The port is set to 8000 (the default used by your python script)

echo "Initializing the vLLM server..."
echo "It will download the weights from HuggingFace the first time (around 14 GB for Mistral-7B)."

python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --host 0.0.0.0 \
  --port 8000
