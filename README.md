# SmolAgents — tiny terminal CodeAgent

A minimal, one-command terminal agent using `smolagents` + `MLXModel`.
It will try to start a local MLX server (if available) and then drop into an interactive REPL so you can keep asking the agent prompts until you type `exit`.

This README gives a quick setup and the common offline options (MLX, `llama.cpp`/GGUF, TGI). Keep it nearby.

---

## What’s in this folder

* `run_agent.py` — interactive REPL that creates a `CodeAgent` using `MLXModel`. Tries to start `mlx_lm.server` automatically.
* `requirements.txt` — Python dependencies.

---

## Quick start (recommended)

1. Create a virtualenv and install:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (Powershell)
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

2. Run the agent:

```bash
python run_agent.py
```

Type prompts at the `>` prompt. Type `exit` or `Ctrl+D` to quit.

---

## Environment / config

You can control behavior with environment variables (or edit `run_agent.py` directly):

* `SMOL_MODEL_ID` — model id / path passed to `MLXModel`. Default: `mlx-community/Qwen2.5-Coder-32B-Instruct-4bit`
* `MLX_EXTRA_ARGS` — extra args forwarded to `mlx_lm.server` when the script auto-starts it (for example `--device cpu`).
* `SMOL_TEMPERATURE`, `SMOL_TOP_K`, `SMOL_TOP_P`, `SMOL_MIN_P`, `SMOL_NUM_CTX` — model init params.

Example:

```bash
SMOL_MODEL_ID="my-local-model" MLX_EXTRA_ARGS="--device cpu" python offline_agents.py
```

---

## Troubleshooting

* **`mlx_lm.server` not found** — either install the MLX package or set `START_MLX_SERVER=False` in `offline_agents.py` and start your server manually.
* **Model too large / out-of-memory** — pick a smaller model (quantized 7B/13B) or use a machine with more RAM / GPU.
* **Slow responses on CPU** — try a quantized model (GGUF 4-bit) or run on GPU.
* **Permissions / PATH issues** — verify your virtualenv is activated and the `mlx_lm.server` binary is on `PATH` (or provide full path).

---

## Example: run FULLY offline with MLX

1. Place model files under `/models/my-model`
   
3. Start server:

```bash
mlx_lm.server --model /models/my-model --device cpu
```

3. Run the agent:

```bash
python offline_agents.py
```

---

## License & credits

Simple demo — use as you like. No warranty. Built with `smolagents`.
