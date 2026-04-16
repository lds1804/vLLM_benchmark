import requests
import time
import threading
import subprocess
import statistics
import argparse
from rich.console import Console
from rich.table import Table

console = Console()

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
    rows = sorted(set(d[row_key] for d in data))
    cols = sorted(set(d[col_key] for d in data))

    table = Table(title=f"\n[bold]{title}[/bold]", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column(str(row_key).capitalize(), style="bold bright_black", justify="center", vertical="middle")
    
    for c in cols:
        table.add_column(f"{col_key}={c}", justify="center", vertical="middle")

    for r in rows:
        row_data = [str(r)]
        for c in cols:
            match = next((d for d in data if d[row_key] == r and d[col_key] == c), None)
            if match:
                cell = f"[bold green]{match['req_per_sec']:.1f} requests/s[/bold green]\n[bold cyan]{match['tok_per_sec']:.0f} tokens/s[/bold cyan]\n[bold yellow]GPU: {match['gpu_avg']:.0f}%[/bold yellow]"
            else:
                cell = "-"
            row_data.append(cell)
        table.add_row(*row_data)

    console.print(table)
    console.print()


def save_markdown_report(engine, results):
    filename = f"benchmark_results_{engine}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Results\n\n")
        f.write(f"**API Engine**: {engine.upper()}\n\n")

        def write_table(title, row_key, col_key):
            f.write(f"## {title}\n\n")
            rows = sorted(set(d[row_key] for d in results))
            cols = sorted(set(d[col_key] for d in results))

            header = f"| {str(row_key).capitalize()} | " + " | ".join([f"{col_key}={c}" for c in cols]) + " |\n"
            separator = "|---|" + "|".join(["---" for _ in cols]) + "|\n"
            f.write(header)
            f.write(separator)

            for r in rows:
                row_str = f"| {r} |"
                for c in cols:
                    match = next((d for d in results if d[row_key] == r and d[col_key] == c), None)
                    if match:
                        cell = f"{match['req_per_sec']:.1f} requests/s<br>{match['tok_per_sec']:.0f} tokens/s<br>GPU: {match['gpu_avg']:.0f}%"
                    else:
                        cell = "-"
                    row_str += f" {cell} |"
                f.write(row_str + "\n")
            f.write("\n")

        write_table("Impact of REQUESTS increase", "requests", "max_tokens")
        write_table("Impact of TOKENS increase", "max_tokens", "requests")
        
    console.print(f"\n[bold green]Results successfully saved in Markdown format to {filename}[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Script")
    parser.add_argument("--engine", type=str, default="vllm", 
                        help="Specify the name of the engine, API, or experiment (e.g. 'vllm', 'hf', 'TGI-Test').")
    args = parser.parse_args()

    results = []

    console.rule(f"[bold blue]Starting Benchmarks (Engine: {args.engine.upper()})[/bold blue]")

    for reqs in REQUEST_LEVELS:
        for max_toks in MAX_TOKENS_LEVELS:
            console.print(f"[bold]Running test:[/bold] [green]{reqs} reqs[/green] | [cyan]max_tokens={max_toks}[/cyan] ...")
            res = run_test(reqs, max_toks)
            results.append(res)

    console.rule("[bold green]ORGANIZED RESULTS[/bold green]")

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

    # Save to markdown file
    save_markdown_report(args.engine, results)

if __name__ == "__main__":
    main()