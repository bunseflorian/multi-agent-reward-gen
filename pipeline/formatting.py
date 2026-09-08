"""Shared formatting helpers for metric history and live logs."""

from __future__ import annotations

import math
from typing import Any

_SUCCESS_METRIC_CANDIDATES = ("final_task_success", "final_success", "mean_success_rate")


def format_metric_history(metric_history: Any) -> str:
    """Format sampled PPO metric history as compact chronological value lists.

    Accepts the list of dicts produced by ``qwen_agentv2.ppo.summary``. Returns
    a compact, single-block string suitable for both Critic prompts and PPO
    summaries.
    """

    if not isinstance(metric_history, list) or not metric_history:
        return "Sampled PPO episode metrics passed to the Critic:\n- none"

    metric_names = sorted(
        {
            key
            for point in metric_history
            if isinstance(point, dict)
            for key in point
            if key != "episode"
        }
    )

    lines = ["Sampled PPO episode metrics passed to the Critic:"]
    for metric_name in metric_names:
        values: list[str] = []
        for point in metric_history:
            if not isinstance(point, dict) or metric_name not in point:
                continue
            values.append(_format_metric_value(point[metric_name]))
        lines.append(f"{metric_name}: [{', '.join(values)}]")
    return "\n".join(lines)


def check_ppo_success_rate(
    metric_history: Any,
    *,
    threshold: float = 0.95,
    window_fraction: float = 0.05,
) -> bool | None:
    """Deterministically decide PPO acceptance from a success-rate metric.

    Looks for a recognized per-episode success metric (``final_task_success``,
    ``final_success``, or ``mean_success_rate``, in that preference order) in
    the sampled ``metric_history`` and computes the success rate over the last
    ``window_fraction`` of its sampled values (at least one point). Returns
    ``True`` when that rate exceeds ``threshold``, ``False`` when it does not,
    and ``None`` when no recognized success metric is present -- callers
    should fall back to LLM judgment in that case (e.g. a PPO runtime crash
    with no metric_history, or a dense-only objective with no success flag).
    """

    if not isinstance(metric_history, list) or not metric_history:
        return None

    metric_name = next(
        (
            name
            for name in _SUCCESS_METRIC_CANDIDATES
            if any(isinstance(point, dict) and name in point for point in metric_history)
        ),
        None,
    )
    if metric_name is None:
        return None

    values = [
        float(point[metric_name])
        for point in metric_history
        if isinstance(point, dict) and isinstance(point.get(metric_name), (int, float, bool))
    ]
    if not values:
        return None

    window_size = max(1, math.ceil(len(values) * window_fraction))
    window = values[-window_size:]
    success_rate = sum(window) / len(window)
    return success_rate > threshold


def format_ppo_live_log(report: Any) -> str:
    """Format a compact PPO outcome block for ``conversation.md``."""

    lines = [
        f"status: `{report.status}`",
        f"passed: `{report.passed}`",
    ]
    if getattr(report, "exit_code", None) is not None:
        lines.append(f"exit_code: `{report.exit_code}`")
    if getattr(report, "timed_out", False):
        lines.append("timed_out: `True`")
    if getattr(report, "log_path", None):
        lines.append(f"log: `{report.log_path}`")
    if getattr(report, "summary_path", None):
        lines.append(f"summary: `{report.summary_path}`")
    errors = getattr(report, "errors", None) or []
    warnings = getattr(report, "warnings", None) or []
    if errors:
        lines.extend(["", "errors:", *[f"- {error}" for error in errors]])
    if warnings:
        lines.extend(["", "warnings:", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines)


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
