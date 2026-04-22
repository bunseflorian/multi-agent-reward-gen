"""Render TensorBoard scalar charts for PPO iterations in the Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd

from ..ppo import extract_episode_metrics_from_event, resolve_latest_experiment_dir


_METRIC_PREFIXES = ("mean_", "max_", "final_")
_ITERATION_COLORS = [
    "#0f766e",
    "#dc2626",
    "#2563eb",
    "#ca8a04",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#65a30d",
]


def render_tensorboard_metrics(st, run_dir: Path, live_event: dict | None = None) -> None:
    """Render all tracked PPO scalar charts for the current run."""

    iteration_metrics = _load_iteration_metrics(run_dir, live_event=live_event)
    if not iteration_metrics:
        return

    metric_names = sorted(
        {
            metric_name
            for metrics in iteration_metrics.values()
            for metric_name in metrics
            if metric_name.startswith(_METRIC_PREFIXES)
        }
    )
    if not metric_names:
        return

    st.subheader("TensorBoard Metrics")
    for metric_name in metric_names:
        chart_frame = _metric_frame(metric_name, iteration_metrics)
        if chart_frame.empty:
            continue
        st.markdown(f"`{metric_name}`")
        chart = (
            alt.Chart(chart_frame)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("episode:Q", title="Episode"),
                y=alt.Y("value:Q", title=metric_name),
                color=alt.Color(
                    "iteration:N",
                    title="Iteration",
                    scale=alt.Scale(
                        domain=sorted(chart_frame["iteration"].unique()),
                        range=_ITERATION_COLORS[: len(chart_frame["iteration"].unique())],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("iteration:N", title="Iteration"),
                    alt.Tooltip("episode:Q", title="Episode"),
                    alt.Tooltip("value:Q", title=metric_name, format=".6g"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)


def _load_iteration_metrics(run_dir: Path, live_event: dict | None = None) -> dict[str, dict[str, list[float]]]:
    iteration_metrics: dict[str, dict[str, list[float]]] = {}
    for summary_path in sorted(run_dir.glob("rl_run_*/ppo_summary.json")):
        iteration_label = summary_path.parent.name.replace("rl_run_", "iteration_")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        event_path = data.get("tensorboard_event_file")
        if not event_path:
            continue
        metrics = _read_episode_scalars(Path(event_path))
        if metrics:
            iteration_metrics[iteration_label] = metrics
    if live_event is not None:
        live_iteration, live_metrics = _load_live_iteration_metrics(live_event)
        if live_iteration and live_metrics:
            iteration_metrics[live_iteration] = live_metrics
    return iteration_metrics


def _read_episode_scalars(event_path: Path) -> dict[str, list[float]]:
    if not event_path.is_file():
        return {}
    metrics = extract_episode_metrics_from_event(event_path)
    return {
        metric_name: values
        for metric_name, values in metrics.items()
        if metric_name.startswith(_METRIC_PREFIXES)
    }


def _load_live_iteration_metrics(live_event: dict) -> tuple[str | None, dict[str, list[float]]]:
    log_path = live_event.get("log_path")
    search_root = live_event.get("experiment_search_root")
    started_at_wall = live_event.get("started_at_wall")
    if not log_path or not search_root:
        return None, {}

    iteration_label = Path(str(log_path)).parent.name.replace("rl_run_", "iteration_")
    experiment_dir = resolve_latest_experiment_dir(
        Path(str(search_root)),
        min_event_mtime=float(started_at_wall) if started_at_wall is not None else None,
    )
    if experiment_dir is None:
        return iteration_label, {}

    summary_dir = experiment_dir / "summaries"
    event_files = sorted(summary_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not event_files:
        return iteration_label, {}
    return iteration_label, _read_episode_scalars(event_files[-1])


def _metric_frame(metric_name: str, iteration_metrics: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for iteration_label, metrics in iteration_metrics.items():
        values = metrics.get(metric_name) or []
        for episode_index, value in enumerate(values, start=1):
            rows.append(
                {
                    "iteration": iteration_label,
                    "episode": episode_index,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)
