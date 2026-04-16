from rich.console import Console
from rich.table import Table

console = Console()

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

mock_results = [
    {"requests": 10, "max_tokens": 50, "req_per_sec": 12.5, "tok_per_sec": 625, "gpu_avg": 45, "gpu_max": 50},
    {"requests": 10, "max_tokens": 100, "req_per_sec": 8.0, "tok_per_sec": 800, "gpu_avg": 55, "gpu_max": 60},
    {"requests": 50, "max_tokens": 50, "req_per_sec": 22.1, "tok_per_sec": 1105, "gpu_avg": 85, "gpu_max": 90},
    {"requests": 50, "max_tokens": 100, "req_per_sec": 15.3, "tok_per_sec": 1530, "gpu_avg": 95, "gpu_max": 100},
]

console.rule("[bold green]ORGANIZED RESULTS (MOCK)[/bold green]")
print_table(
    "Impact of REQUESTS increase",
    mock_results,
    row_key="requests",
    col_key="max_tokens"
)
