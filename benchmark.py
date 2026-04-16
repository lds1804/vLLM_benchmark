import requests
import time
import threading
import subprocess
import statistics

URL = "http://localhost:8000/v1/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# testing parameters
REQUEST_LEVELS = [10, 50, 100]
MAX_TOKENS_LEVELS = [50, 100, 200]

gpu_usage_samples = []
running = True


def monitor_gpu():
    global gpu_usage_samples, running
    while running:
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            )
            usage = int(result.decode().strip().split("\n")[0])
            gpu_usage_samples.append(usage)
        except:
            pass
        time.sleep(0.5)


def send_request(max_tokens):
    payload = {
        "model": MODEL,
        "prompt": "Explain parallelism in distributed systems",
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            tokens = data["usage"]["completion_tokens"]
            return tokens
    except:
        pass
    return 0


def run_test(num_requests, max_tokens):
    global gpu_usage_samples, running
    gpu_usage_samples = []
    running = True

    threads = []
    results = []

    gpu_thread = threading.Thread(target=monitor_gpu)
    gpu_thread.start()

    start = time.time()

    def worker():
        tokens = send_request(max_tokens)
        results.append(tokens)

    for _ in range(num_requests):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    end = time.time()
    running = False
    gpu_thread.join()

    duration = end - start
    total_tokens = sum(results)

    req_per_sec = num_requests / duration
    tok_per_sec = total_tokens / duration if duration > 0 else 0

    gpu_avg = statistics.mean(gpu_usage_samples) if gpu_usage_samples else 0
    gpu_max = max(gpu_usage_samples) if gpu_usage_samples else 0

    return {
        "requests": num_requests,
        "max_tokens": max_tokens,
        "duration": duration,
        "req_per_sec": req_per_sec,
        "tok_per_sec": tok_per_sec,
        "gpu_avg": gpu_avg,
        "gpu_max": gpu_max
    }


def print_table(title, data, row_key, col_key):
    print(f"\n=== {title} ===\n")

    rows = sorted(set(d[row_key] for d in data))
    cols = sorted(set(d[col_key] for d in data))

    # build cells
    table = {}
    max_cell_width = 0

    for r in rows:
        for c in cols:
            match = next(
                (d for d in data if d[row_key] == r and d[col_key] == c),
                None
            )
            if match:
                cell = f"{match['req_per_sec']:.1f} r/s | {match['tok_per_sec']:.0f} t/s | {match['gpu_avg']:.0f}%"
            else:
                cell = "-"
            table[(r, c)] = cell
            max_cell_width = max(max_cell_width, len(cell))

    col_width = max_cell_width + 2
    row_label_width = max(len(str(row_key)), max(len(str(r)) for r in rows)) + 2

    # MAIN HEADER (centered to the column width)
    header = str(row_key).ljust(row_label_width)
    for c in cols:
        col_name = f"{col_key}={c}"
        header += col_name.center(col_width)
    print(header)

    # SUB-HEADER (same logic)
    sub_header = " ".ljust(row_label_width)
    metric_label = "req/s | tok/s | GPU%"
    for _ in cols:
        sub_header += metric_label.center(col_width)
    print(sub_header)

    print("-" * (row_label_width + col_width * len(cols)))

    # rows (keeps consistent alignment)
    for r in rows:
        line = str(r).ljust(row_label_width)
        for c in cols:
            line += table[(r, c)].ljust(col_width)
        print(line)


def main():
    results = []

    for reqs in REQUEST_LEVELS:
        for max_toks in MAX_TOKENS_LEVELS:
            print(f"\nRunning test: {reqs} reqs | max_tokens={max_toks}")
            res = run_test(reqs, max_toks)
            results.append(res)

    print("\n\n========== ORGANIZED RESULTS ==========\n")

    # Table 1: requests impact
    print_table(
        "Impact of REQUESTS increase",
        results,
        row_key="requests",
        col_key="max_tokens"
    )

    # Table 2: tokens impact (inverted)
    print_table(
        "Impact of TOKENS increase",
        results,
        row_key="max_tokens",
        col_key="requests"
    )


if __name__ == "__main__":
    main()