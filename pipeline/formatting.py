"""Shared formatting helpers for metric history and live logs."""

from __future__ import annotations

from typing import Any


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
