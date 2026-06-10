"""Focused tests for central config loading and run recording."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_IMPORT_ROOT))

from qwen_agentv2.configs import build_ppo_config, clear_app_config_cache, load_app_config
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


def test_run_recorder_writes_conversation(tmp_path):
    recorder = RunRecorder(tmp_path)

    recorder.write("hello from planner", agent="planner", kind="input")
    recorder.write("plan result", agent="planner", kind="output")

    conversation = (tmp_path / "conversation.md").read_text(encoding="utf-8")

    assert "## Planner — Input" in conversation
    assert "hello from planner" in conversation
    assert "## Planner — Output" in conversation
    assert "plan result" in conversation


def test_run_recorder_null_logger_does_nothing(tmp_path):
    from qwen_agentv2.pipeline.logger import null_logger
    recorder = null_logger()
    recorder.write("should not crash", agent="planner", kind="input")
