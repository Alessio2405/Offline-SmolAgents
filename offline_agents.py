import os
import sys
import time
import shlex
import signal
import subprocess
from pathlib import Path

try:
    from smolagents import CodeAgent, MLXModel
except Exception as e:
    print("Error importing smolagents. Make sure you installed requirements.txt")
    raise

from rich.console import Console
from rich.markdown import Markdown

console = Console()

MODEL_ID = os.environ.get("SMOL_MODEL_ID", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2")
MLX_EXTRA_ARGS = os.environ.get("MLX_EXTRA_ARGS", "")  # e.g. "--device cuda --quantize 4bit"
START_MLX_SERVER = True  # script will try to start mlx_lm.server if available
MLX_SERVER_CMD = f"mlx_lm.server --model {shlex.quote(MODEL_ID)} {MLX_EXTRA_ARGS}".strip()

SERVER_WAIT_SECONDS = 5

def start_mlx_server():
    """Attempt to start an mlx_lm.server subprocess (detached) and return Popen or None."""
    try:
        parts = shlex.split(MLX_SERVER_CMD)
        console.print(f"[bold yellow]Starting MLX server:[/bold yellow] {MLX_SERVER_CMD}")
        proc = subprocess.Popen(parts, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        console.print("[green]Waiting for server to become ready...[/green]")
        while time.time() - t0 < SERVER_WAIT_SECONDS:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            console.print(f"[dim]{line.rstrip()}[/dim]")
            if "Running" in line or "listening" in line.lower() or "started" in line.lower():
                break
        return proc
    except FileNotFoundError:
        console.print("[red]mlx_lm.server binary not found in PATH. Skipping auto-start.[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Failed to start MLX server: {e}[/red]")
        return None

def stop_proc(proc):
    if not proc:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass

def main():
    console.rule("[bold blue]SmolAgent REPL (CodeAgent + MLXModel)[/bold blue]")
    console.print(Markdown(f"**Model:** `{MODEL_ID}`"))
    mlx_proc = None
    if START_MLX_SERVER:
        mlx_proc = start_mlx_server()
        if mlx_proc is None:
            console.print("[yellow]Proceeding without local MLX server — ensure you have another model endpoint available.[/yellow]")

    console.print("[green]Initializing MLXModel and CodeAgent...[/green]")
    try:
        model = MLXModel(
            model_id=MODEL_ID,
            temperature=float(os.environ.get("SMOL_TEMPERATURE", 0.7)),
            top_k=int(os.environ.get("SMOL_TOP_K", 20)),
            top_p=float(os.environ.get("SMOL_TOP_P", 0.8)),
            min_p=float(os.environ.get("SMOL_MIN_P", 0.05)),
            num_ctx=int(os.environ.get("SMOL_NUM_CTX", 32768)), 
            max_tokens=8192,
            # trust_remote_code=True might be necessary for some HF-converted models, uncomment if you have issues
        )
        agent = CodeAgent(tools=[], model=model, add_base_tools=True)
    except Exception as e:
        console.print(f"[red]Failed to initialize model/agent: {e}[/red]")
        stop_proc(mlx_proc)
        sys.exit(1)

    console.print("[bold]Ready. Type prompts (or 'exit' to quit).[/bold]")

    try:
        while True:
            try:
                prompt = console.input("[cyan]> [/cyan]")
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt:
                continue
            if prompt.strip().lower() in ("exit", "quit"):
                break

            try:
                console.print("[blue]Running agent...[/blue]")
                out = agent.run(prompt)
                console.print(Markdown(f"**Agent output:**\n\n```\n{out}\n```"))
            except Exception as e:
                console.print(f"[red]Agent run failed: {e}[/red]")

    finally:
        console.print("[yellow]Shutting down...[/yellow]")
        stop_proc(mlx_proc)
        console.print("[green]Goodbye![/green]")

if __name__ == "__main__":
    main()