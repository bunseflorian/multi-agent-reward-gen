# Qwen Agent v2 — Automated Reward Code Generator for IsaacLab

A multi-agent pipeline that automatically generates and validates reward functions for quadcopter reinforcement learning environments in [IsaacLab](https://isaac-sim.github.io/IsaacLab/). A local LLM served via [vLLM](https://github.com/vllm-project/vllm) powers four specialized agents (Planner, Coder, Reviewer, Critic) that iteratively write and evaluate reward code against a PPO training loop.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [LLM Server Setup](#llm-server-setup)
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
 run artifacts
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU with CUDA | Required by IsaacLab for simulation and PPO training |
| [IsaacLab](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html) | Must be installed in a conda environment (default name: `isaaclab`) |
| Conda / Miniconda | Used to manage the conda environments |
| Python 3.10 – 3.11 | Required by IsaacLab and vLLM |
| vLLM server | Runs in a separate conda environment — on the same machine or a dedicated server PC |

---

## Installation

### 1. IsaacLab environment

Follow the [official IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). The environment can have any name, but if it differs from `isaaclab` update `paths.isaaclab_conda_env` in [`configs/app.yaml`](configs/app.yaml).

Install the task package and UI dependencies (from the IsaacLab root):
```bash
conda activate isaaclab
pip install -e source/isaaclab_tasks
pip install -r source/isaaclab_tasks/isaaclab_tasks/direct/qwen_agentv2/requirements-isaaclab.txt
```

### 2. vLLM server environment

```bash
conda create -n qwen-vllm-server python=3.11 -y
conda activate qwen-vllm-server
pip install -r qwen_vllm_server/requirements.txt
```

---

## LLM Server Setup

The vLLM server is managed via [`qwen_vllm_server/`](qwen_vllm_server/), which is self-contained and can run on the same machine or be copied to a dedicated server PC.

### Download the model

```bash
conda activate qwen-vllm-server
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='<org>/<model-name>',        # e.g. Qwen/Qwen3.5-4B
    local_dir='qwen_vllm_server/models/<model-dir>'
)
"
```

Update `model_path` in [`qwen_vllm_server/config.yaml`](qwen_vllm_server/config.yaml) to match the downloaded directory.

### Start the server

```bash
conda activate qwen-vllm-server
python qwen_vllm_server/launch_server.py
```

Alternatively, use the **Start server** button in the UI sidebar (see [Usage](#usage)).

Verify the server is running:
```bash
curl http://localhost:8000/v1/models
```

### Running on a separate machine

Copy `qwen_vllm_server/` to the server PC, install the requirements there, and update `llm.base_url` in `configs/app.yaml` to point to the server's IP:

```bash
scp -r qwen_vllm_server/ user@server-pc:~/
# On the server PC: edit config.yaml, then python launch_server.py
```

---

## Configuration

All runtime settings are in [`configs/app.yaml`](configs/app.yaml). The most important fields:

```yaml
paths:
  isaaclab_root: /path/to/IsaacLab       # absolute path to your IsaacLab checkout

llm:
  base_url: http://localhost:8000/v1     # URL of the running vLLM server
  model: <org>/<model-name>              # must match served_model_name in qwen_vllm_server/config.yaml

server:
  launcher: source/.../qwen_vllm_server/launch_server.py  # enables Start button + auto pause/resume
  pid_file: source/.../qwen_vllm_server/.qwen_server.pid
  startup_timeout_seconds: 300

ppo:
  run_rl_training: false                 # set true to enable PPO validation
```

**`server.launcher` / `server.pid_file`** enable two features:
- **Start / Stop** buttons in the UI sidebar.
- Automatic server kill before PPO training and restart afterward (needed when server and RL share one GPU). The server is killed with SIGKILL so GPU memory is freed immediately; PPO's own initialisation time (~60–90 s) gives the OS sufficient time to reclaim it. Leave both fields empty when the server runs on a separate machine.

Any config value can also be overridden via `QWEN_*` environment variables:

| Variable | Config key |
|---|---|
| `QWEN_BASE_URL` | `llm.base_url` |
| `QWEN_MODEL` | `llm.model` |
| `QWEN_ISAACLAB_ROOT` | `paths.isaaclab_root` |
| `QWEN_SERVER_LAUNCHER` | `server.launcher` |
| `QWEN_SERVER_PID_FILE` | `server.pid_file` |
| `QWEN_RUN_RL_TRAINING` | `ppo.run_rl_training` |

---

## Usage

### Run the Streamlit UI

```bash
conda activate isaaclab
cd /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/qwen_agentv2
streamlit run ui/app.py
```

The UI opens in your browser at `http://localhost:8501`.

**Sidebar (left panel):**
- Set the **Server URL** (defaults to `llm.base_url` from `configs/app.yaml`).
- A green indicator confirms the server is reachable.
- **Start / Stop** buttons appear when `server.launcher` is configured. Start is disabled while the server is running, Stop while it is not.

**Main panel:**
1. Select an **Environment file** from the `environments/` directory.
2. Enter a plain-text **User Prompt** describing the reward objective.
3. Toggle **Run RL training** to enable PPO validation after static review.
4. Adjust iteration budgets and PPO settings as needed.
5. Click **Generate reward code**.

Results appear inline and are saved under `runs/`.

> **Shared GPU (server and RL training on the same machine):** When `server.launcher` is configured, the UI automatically kills the vLLM server (SIGKILL) before each PPO run and restarts it afterward. No manual intervention needed.

---

## Project Structure

```
qwen_agentv2/
├── agents/             # PPO hyperparameter configs (rl_games, skrl, rsl_rl)
├── configs/            # Central app config (app.yaml + loader)
├── core/               # Shared dataclasses (TaskBrief, PPOConfig, RunState, …)
├── environments/       # IsaacLab quadcopter environments (template + variants)
├── llm/                # Qwen API client and config loader
├── pipeline/           # Agent implementations + orchestrator
│   ├── orchestrator.py     Runs the full Planner→Coder→Reviewer→Tester→Critic loop
│   ├── planner.py
│   ├── coder.py
│   ├── reviewer.py
│   ├── tester.py
│   ├── critic.py
│   ├── prompts.py          All system and user prompt templates
│   └── logger.py           RunRecorder — appends all agent turns to conversation.md
├── ppo/                # PPO subprocess runner
├── qwen_vllm_server/   # Standalone vLLM server launcher (config.yaml + launch_server.py)
├── runs/               # Generated run directories (created at runtime)
├── server/             # Server health check and stop/start control
├── test/               # Agent smoke tests
└── ui/                 # Streamlit web interface
```

**Run directory contents** (created per generation run under `runs/`):
```
runs/<timestamp>_<status>/
├── conversation.md         all agent turns (planner, coder, reviewer, critic)
├── ppo.log                 (if PPO ran)
└── ppo_summary.json        (if PPO ran)
```
