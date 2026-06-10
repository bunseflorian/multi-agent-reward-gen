# Qwen Agent v2 — Automated Reward Code Generator for IsaacLab

A multi-agent pipeline that automatically generates and validates reward functions for quadcopter reinforcement learning environments in [IsaacLab](https://isaac-sim.github.io/IsaacLab/). A local LLM served via [vLLM](https://github.com/vllm-project/vllm) powers four specialized agents (Planner, Coder, Reviewer, Critic) that iteratively write and evaluate reward code against a PPO training loop.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Run the Streamlit UI](#run-the-streamlit-ui)
- [Project Structure](#project-structure)

---

## Overview

Given a plain-text reward objective (e.g. *"Minimize distance to the target while maintaining low angular velocity"*), the system:

1. **Plans** a concrete reward recipe from the environment's source code.
2. **Codes** the `_compute_generated_reward` body.
3. **Reviews** the code with a static AST/mock-execution check followed by an LLM compliance pass.
4. **Tests** the reward in a real IsaacLab PPO run (optional).
5. **Critiques** the training metrics and decides to accept or request a revision.

The loop repeats until the reward is accepted or the iteration budget is exhausted. All artifacts (plan, code, conversation log, PPO metrics) are saved in a timestamped run directory.

---

## Architecture

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    Orchestrator                       │
│                                                      │
│  Planner ──► Coder ──► Reviewer ──► Tester ──► Critic│
│                  ▲___________feedback_______________|  │
└──────────────────────────────────────────────────────┘
    │
    ▼
 final_reward.py  +  run artifacts
```

See [docs/architecture.md](docs/architecture.md) for the full UML sequence diagram and per-agent input/output specs.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU with CUDA | vLLM requires CUDA; VRAM requirement depends on the chosen model |
| [IsaacLab](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html) | Must be installed in a conda environment (default name: `isaaclab`) |
| Conda / Miniconda | Used to manage the two separate environments |
| Python 3.10 – 3.11 | Required by both IsaacLab and vLLM |

---

## Installation

The project requires **two separate conda environments**: one for IsaacLab (simulation + PPO) and one for the vLLM server (LLM inference). They must stay separate because their CUDA and Python dependencies conflict.

### 1. IsaacLab environment

Follow the [official IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). The environment can have any name, but if it differs from `isaaclab` you must update `paths.isaaclab_conda_env` in [`configs/app.yaml`](configs/app.yaml#L15).

Verify the installation:
```bash
conda activate isaaclab
python -c "import isaaclab; print('IsaacLab OK')"
```

Install the task package (from the IsaacLab root):
```bash
conda activate isaaclab
pip install -e source/isaaclab_tasks
```

Install the UI and pipeline dependencies:
```bash
conda activate isaaclab
pip install -r source/isaaclab_tasks/isaaclab_tasks/direct/qwen_agentv2/requirements-isaaclab.txt
```

### 2. vLLM server environment

Create a dedicated environment for the LLM server:
```bash
conda create -n qwen-vllm-server python=3.11 -y
conda activate qwen-vllm-server
pip install -r source/isaaclab_tasks/isaaclab_tasks/direct/qwen_agentv2/requirements-server.txt
```

Verify:
```bash
conda activate qwen-vllm-server
python -c "import vllm; print('vLLM OK')"
```

---

## Model Setup

Download any vLLM-compatible chat model (e.g. from Hugging Face) and place it in the `models/` directory.
The `models/` directory is **git-ignored** — model weights are never committed to the repository.

```bash
conda activate qwen-vllm-server
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='<org>/<model-name>',   # e.g. Qwen/Qwen3.5-4B
    local_dir='models/<model-dir>'  # e.g. models/llm-qwen3.5-4B
)
"
```

Confirm that `models/<model-dir>/config.json` exists after the download, then update `paths.qwen_model` and `llm.model` in [configs/app.yaml](configs/app.yaml) to match.

---

## Configuration

All runtime settings are in [configs/app.yaml](configs/app.yaml). The most important fields:

```yaml
paths:
  isaaclab_root: /path/to/IsaacLab          # absolute path to your IsaacLab checkout
  qwen_model: source/isaaclab_tasks/isaaclab_tasks/direct/qwen_agentv2/models/<model-dir>

llm:
  base_url: http://localhost:8000/v1         # vLLM server URL
  model: <org>/<model-name>                  # must match --served-model-name of the server

ppo:
  run_rl_training: false                     # set true to enable PPO validation
  num_envs: 4096
  max_iterations: 150
  timeout_seconds: 300
```

**Update `paths.isaaclab_root`** to match your local IsaacLab installation path before running.

Any config value can also be overridden at runtime via `QWEN_*` environment variables (see [Environment Override Variables](#environment-override-variables)).

---

## Usage

### Run the Streamlit UI

Activate the `isaaclab` environment and launch the web interface:

```bash
conda activate isaaclab
cd /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/qwen_agentv2
streamlit run ui/app.py
```

The UI opens in your browser at `http://localhost:8501`.

**Sidebar (left panel):**
- Configure the vLLM server host, port, model path, and GPU settings.
- Use **Start server** to launch the vLLM process (runs in the `qwen-vllm-server` conda environment) and **Stop server** to shut it down.
- A green indicator confirms the server is reachable before generating.

**Main panel:**
1. Select an **Environment file** from the `environments/` directory.
2. Enter a plain-text **User Prompt** describing the reward objective.
3. Toggle **Run RL training** to enable PPO validation after static review.
4. Adjust iteration budgets and PPO settings as needed.
5. Click **Generate reward code**.

Results (plan, generated code, conversation log) appear inline and are saved under `runs/`.

---

## Project Structure

```
qwen_agentv2/
├── agents/             # PPO hyperparameter configs (rl_games, skrl, rsl_rl)
├── cli/                # CLI entry point and display helpers
├── configs/            # Central app config (app.yaml + loader)
├── core/               # Shared dataclasses (TaskBrief, PPOConfig, RunState, …)
├── docs/               # Architecture documentation
├── environments/       # IsaacLab quadcopter environments (template + variants)
├── llm/                # Qwen API client and config loader
├── models/             # Local model weights (git-ignored, not committed)
├── pipeline/           # Agent implementations + orchestrator
│   ├── orchestrator.py     Runs the full Planner→Coder→Reviewer→Tester→Critic loop
│   ├── planner.py
│   ├── coder.py
│   ├── reviewer.py
│   ├── tester.py
│   ├── critic.py
│   ├── prompts.py          All system and user prompt templates
│   └── logger.py           RunRecorder — writes events.jsonl and run artifacts
├── ppo/                # PPO subprocess runner
├── runs/               # Generated run directories (created at runtime)
├── server/             # vLLM server launcher and health check
├── test/               # Agent smoke tests
└── ui/                 # Streamlit web interface
```

**Run directory contents** (created per generation run under `runs/`):
```
runs/<timestamp>_<status>/
├── task_brief.md
├── env_code_context.md
├── conversation.md
├── full_conversation.md
├── events.jsonl
├── run_state.json
├── final_reward.py         (if a candidate was accepted)
├── ppo.log                 (if PPO ran)
└── ppo_summary.json        (if PPO ran)
```

