from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import torch

app = FastAPI()

MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

print(f"Loading Hugging Face model {MODEL}...")
# Device map auto will distribute the model across available GPUs automatically.
# Requires 'accelerate' library.
pipe = pipeline("text-generation", model=MODEL, device_map="auto", torch_dtype=torch.float16)
print("Model loaded successfully!")

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 50

@app.post("/v1/completions")
def completions(request: CompletionRequest):
    # Generate text based on the prompt
    outputs = pipe(
        request.prompt, 
        max_new_tokens=request.max_tokens, 
        do_sample=True, 
        return_full_text=False
    )
    
    generated_text = outputs[0]["generated_text"]
    
    # Return in exactly the same format as vLLM / OpenAI to keep benchmark.py compatible
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "model": request.model,
        "choices": [
            {
                "text": generated_text,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": request.max_tokens,
            "total_tokens": request.max_tokens
        }
    }

if __name__ == "__main__":
    # Ensure it binds to port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
