"""Focused tests for central config loading and run recording."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_IMPORT_ROOT))

from qwen_agentv2.configs import build_ppo_config, clear_app_config_cache, load_app_config
from qwen_agentv2.core import CodeArtifact, RunState, RunStatus, TaskBrief, TestReport
from qwen_agentv2.pipeline import RunRecorder


def test_load_app_config_applies_environment_overrides(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("QWEN_PPO_NUM_ENVS", "16")
    clear_app_config_cache()

    config = load_app_config()

    assert config["llm"]["base_url"] == "http://127.0.0.1:9999/v1"
    assert config["ppo"]["num_envs"] == 16
    assert build_ppo_config().num_envs == 16

    clear_app_config_cache()


def test_run_recorder_writes_events_and_run_state(tmp_path):
    state = RunState(
        task_brief=TaskBrief(
            task_id="unit_task",
            objective="test objective",
            env_source_path="/tmp/env.py",
            env_code_context="class Env: pass",
        ),
        status=RunStatus.ACCEPTED_STATIC,
        final_code=CodeArtifact(code="def reward():\n    return 1", raw_response="raw"),
        final_report=TestReport(passed=True),
    )
    recorder = RunRecorder(tmp_path)
    state.recorder = recorder

    recorder.event("run_started", task_id="unit_task")
    recorder.log_full("prompt", agent="planner", kind="input")
    recorder.log_output("answer", agent="planner", extract=False)
    recorder.save_run_state(state)

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    run_state = json.loads((tmp_path / "run_state.json").read_text())

    assert events[0]["event"] == "run_started"
    assert any(event["event"] == "agent_prompt" and event["content"] == "prompt" for event in events)
    assert any(event["event"] == "conversation_entry" and event["content"] == "answer" for event in events)
    assert any(event["event"] == "run_state" for event in events)
    assert (tmp_path / "conversation.md").is_file()
    assert (tmp_path / "full_conversation.md").is_file()
    assert run_state["status"] == "accepted_static"
    assert run_state["task_brief"]["task_id"] == "unit_task"
    assert run_state["final_code"]["code"].startswith("def reward")
