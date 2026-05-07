"""Focused checks for reward-term sum logging support."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_IMPORT_ROOT))

from qwen_agentv2.pipeline.static_check import static_check


def test_static_check_requires_reward_component_sums():
    code = """
x = self._x
reward_x = x
reward = reward_x
"""

    report = static_check(code, allowed_tensors=["self._x [shape: (N,)]"])

    assert not report.passed
    assert any("Reward component logging is incomplete" in error for error in report.errors)


def test_static_check_accepts_reward_component_sums():
    code = """
x = self._x
reward_x = x
reward = reward_x
self._episode_reward_term_sums.setdefault("reward_term_reward_x", torch.zeros_like(reward_x))
self._episode_reward_term_sums["reward_term_reward_x"] += reward_x
"""

    report = static_check(code, allowed_tensors=["self._x [shape: (N,)]"])

    assert not report.passed
    assert any("final_reward_term_reward_x" in error for error in report.errors)


def test_static_check_accepts_reward_terms_in_final_values():
    code = """
reward_x = self._x
reward = reward_x
self._episode_final_values.setdefault("final_reward_term_reward_x", torch.zeros_like(reward_x))
self._episode_final_values["final_reward_term_reward_x"] += reward_x
"""

    report = static_check(code, allowed_tensors=["self._x [shape: (N,)]"])

    assert report.passed
