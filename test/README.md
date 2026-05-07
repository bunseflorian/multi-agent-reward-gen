# Agent smoke tests

Quick probes for each agent's system prompt against the running vLLM server — no IsaacLab, no PPO, no orchestrator.

Each test:
1. Extracts the agent's system prompt from [pipeline/prompts.py](../pipeline/prompts.py) via regex (avoids loading `isaaclab_tasks`).
2. Builds a realistic user prompt from a hard-coded fixture (real env file, real plan text, real PPO metrics).
3. Hits the local vLLM server at `http://localhost:8000/v1` and prints the raw response.

Only the standard library is used (`urllib`, `json`, `re`). No `openai` / `httpx` / `toml` deps.

## Prerequisites

The vLLM server must be running locally on port 8000 serving `Qwen/Qwen3.5-4B`. Start it from the UI sidebar or via [server/launcher.py](../server/launcher.py).

## Usage

```bash
python3 test/test_planner.py    # env file + user prompt → plan artifact
python3 test/test_coder.py      # plan → reward-code body
python3 test/test_reviewer.py   # plan + code + static report → PASS/FAIL
python3 test/test_critic.py     # plan + code + PPO metrics → ACCEPT/REVISE
```

Each script prints its output to stdout. Override the fixtures by editing the constants at the top of each file.

## Files

- [common.py](common.py) — `load_prompt(name)` and `call_llm(system, user, ...)`.
- [test_planner.py](test_planner.py) — reads [environments/quadcopter_env_tracking.py](../environments/quadcopter_env_tracking.py) verbatim as env context.
- [test_coder.py](test_coder.py) — fresh-write path (`CODER_INITIAL_PROMPT + CODER_RULES_PROMPT`); adapt to test the feedback path by switching to `CODER_FEEDBACK_PROMPT` and adding reviewer/critic feedback blocks.
- [test_reviewer.py](test_reviewer.py) — toggle `STATIC_REPORT_PASSED` and `STATIC_ERRORS` to test both static-fail and static-pass branches.
- [test_critic.py](test_critic.py) — canonical "on-track" quadcopter trajectory (final_success trends 0 → 1, dist converges 1.7 → 0.2). Expected: `DECISION: ACCEPT, FIX: none`.
