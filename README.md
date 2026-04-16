# LLM Inference Benchmark: vLLM vs Hugging Face

This repository contains a complete benchmark suite (stress testing) designed to empirically evaluate and compare the inference performance of the ultra-high performance library **vLLM** against the vanilla implementation of **Hugging Face Transformers**.

The goal of this suite is to provide data that proves the stark processing difference provided by optimized cache management (PagedAttention) and high-level parallelism, displaying metrics such as RPS (Requests Per Second), TPS (Tokens Per Second), and GPU Utilization.

## 🛠️ What's inside?

- `install_vllm.sh` & `requirements.txt`: Installs all environment dependencies.
- `start_server.sh`: Starts the native vLLM engine hosting an OpenAI-compatible API on port 8000.
- `hf_api_server.py` & `start_hf_server.sh`: Spins up a mock server using FastAPI and the Hugging Face pipeline to simulate the exact OpenAI API path (ensuring a rigorous and fair 1:1 comparison).
- `benchmark.py`: The star of the project. A multi-threading orchestrator that bombards the server while simultaneously measuring your GPU utilization by running `nvidia-smi` in the background.

---

## 🚀 How to Run

### 1. Environment Setup
The only necessary setup step is to run the installer (we recommend a Linux environment or a Vast.ai container):
```bash
bash install_vllm.sh
```

### 2. Start the Target Engine (Background)
In a terminal window, start the server of your choice:

To test the revolutionary optimizations of **vLLM**:
```bash
bash start_server.sh
```

Or, if you want to run on **Hugging Face** to get your baseline:
```bash
bash start_hf_server.sh
```
*(Note: The server will be exposed at `http://localhost:8000`)*

### 3. Trigger the Benchmark Load

With the server running, open a new shell/terminal and run the test. You can (and should) use the `--engine` parameter to categorize the name of your output results.

To run with the vLLM engine:
```bash
python benchmark.py --engine VLLM
```

To run against the original vanilla HF implementation:
```bash
python benchmark.py --engine HuggingFace-Original
```

---

## 📊 Reports and Outputs

The results will be displayed immediately and beautifully in your terminal with natively built tables by Python's `rich` library. The output will show the impact that increasing **Concurrency (`requests`)** and **Output Depth (`max_tokens`)** has on RPS, TPS, and Cost (% of hardware GPU utilization).

At the end of each run, the suite will automatically generate and save an elegant Markdown visual artifact in the root directory:
`benchmark_results_VLLM.md`

Use this project on heavy GPU instances and show the world your empirical evidence!
