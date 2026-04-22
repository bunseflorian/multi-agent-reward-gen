"""Summarize PPO log output and TensorBoard scalars for the Critic prompt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..pipeline.formatting import format_metric_history
from .command import configured_conda_env


MAX_METRIC_HISTORY_POINTS = 40


def summarize_ppo_log(
    log_path: str | Path,
    *,
    exit_code: int | None = None,
    timed_out: bool = False,
    experiment_dir: str | Path | None = None,
    experiment_search_root: str | Path | None = None,
    min_event_mtime: float | None = None,
) -> dict:
    """Return a compact summary of a PPO run: errors, tail, metric history."""

    path = Path(log_path)
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    lines = text.splitlines()

    resolved_experiment_dir = _resolve_experiment_dir(
        experiment_dir=experiment_dir,
        experiment_search_root=experiment_search_root,
        min_event_mtime=min_event_mtime,
    )

    metric_values: dict[str, list[float]] = {}
    latest_event_file: Path | None = None
    if resolved_experiment_dir is not None:
        latest_event_file = _latest_tensorboard_event_file(resolved_experiment_dir / "summaries")
        metric_values = _extract_all_episode_metrics(resolved_experiment_dir)

    metric_history = _sample_metric_history(metric_values, MAX_METRIC_HISTORY_POINTS)

    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "line_count": len(lines),
        "errors": _select_lines(lines, ("error", "exception", "traceback", "failed")),
        "tail": lines[-40:],
        "experiment_dir": str(resolved_experiment_dir) if resolved_experiment_dir is not None else None,
        "tensorboard_event_file": str(latest_event_file) if latest_event_file is not None else None,
        "metric_history_sampling": {
            "strategy": "all_if_short_else_evenly_spaced",
            "max_points": MAX_METRIC_HISTORY_POINTS,
            "source_points": max((len(values) for values in metric_values.values()), default=0),
        },
        "metric_history": metric_history,
        "critic_metric_history": format_metric_history(metric_history),
    }


def resolve_latest_experiment_dir(
    experiment_search_root: str | Path,
    *,
    min_event_mtime: float | None = None,
) -> Path | None:
    """Return the newest experiment dir with TensorBoard events."""

    return _find_latest_experiment_dir(experiment_search_root, min_event_mtime=min_event_mtime)


def extract_episode_metrics_from_event(event_path: str | Path) -> dict[str, list[float]]:
    """Return ``Episode/*`` scalars from a TensorBoard event file."""

    return _extract_episode_metrics_in_process(Path(event_path))


# ---------------------------------------------------------------------------
# Experiment-dir resolution
# ---------------------------------------------------------------------------

def _resolve_experiment_dir(
    *,
    experiment_dir: str | Path | None,
    experiment_search_root: str | Path | None,
    min_event_mtime: float | None,
) -> Path | None:
    if experiment_dir is not None:
        path = Path(experiment_dir)
        if _experiment_has_tensorboard_events(path):
            return path
        if path.is_dir():
            fallback = _find_latest_experiment_dir(path, min_event_mtime=min_event_mtime)
            if fallback is not None:
                return fallback
    if experiment_search_root is not None:
        return _find_latest_experiment_dir(experiment_search_root, min_event_mtime=min_event_mtime)
    return None


def _find_latest_experiment_dir(
    experiment_search_root: str | Path,
    *,
    min_event_mtime: float | None = None,
) -> Path | None:
    """Return the newest experiment directory containing TensorBoard events.

    Why: RL-Games output via ``conda run`` can be buffered and timeout before the
    experiment path is flushed into ``ppo.log``, but the events file is written
    directly, so we fall back to it to extract metrics.
    """

    root = Path(experiment_search_root)
    if not root.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for event_file in root.glob("*/summaries/events.out.tfevents.*"):
        mtime = event_file.stat().st_mtime
        if min_event_mtime is not None and mtime < min_event_mtime:
            continue
        candidates.append((mtime, event_file.parent.parent))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _experiment_has_tensorboard_events(experiment_dir: Path) -> bool:
    summary_dir = experiment_dir / "summaries"
    return summary_dir.is_dir() and any(summary_dir.glob("events.out.tfevents.*"))


def _latest_tensorboard_event_file(summary_dir: Path) -> Path | None:
    if not summary_dir.is_dir():
        return None
    event_files = list(summary_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    return max(event_files, key=lambda path: path.stat().st_mtime)


# ---------------------------------------------------------------------------
# TensorBoard scalar extraction
# ---------------------------------------------------------------------------

def _extract_all_episode_metrics(experiment_dir: Path) -> dict[str, list[float]]:
    """Return ``Episode/*`` scalars from the newest event file in ``summaries/``."""

    summary_dir = experiment_dir / "summaries"
    if not summary_dir.is_dir():
        return {}
    latest_event_file = _latest_tensorboard_event_file(summary_dir)
    if latest_event_file is None:
        return {}
    result = _extract_episode_metrics_in_process(latest_event_file)
    if result:
        return result
    return _extract_episode_metrics_with_conda(latest_event_file)


def _extract_episode_metrics_in_process(event_path: Path) -> dict[str, list[float]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator  # type: ignore
    except ImportError:
        return {}
    accumulator = event_accumulator.EventAccumulator(str(event_path))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    result: dict[str, list[float]] = {}
    for tag in tags:
        if not tag.startswith("Episode/"):
            continue
        key = tag[len("Episode/") :]
        result[key] = [event.value for event in accumulator.Scalars(tag)]
    return result


def _extract_episode_metrics_with_conda(event_path: Path) -> dict[str, list[float]]:
    conda_env = configured_conda_env()
    if not conda_env:
        return {}
    script = (
        "import json, sys\n"
        "from tensorboard.backend.event_processing import event_accumulator\n"
        "event_path = sys.argv[1]\n"
        "acc = event_accumulator.EventAccumulator(event_path)\n"
        "acc.Reload()\n"
        "tags = acc.Tags().get('scalars', [])\n"
        "result = {}\n"
        "for tag in tags:\n"
        "    if not tag.startswith('Episode/'):\n"
        "        continue\n"
        "    key = tag[len('Episode/'):]\n"
        "    result[key] = [e.value for e in acc.Scalars(tag)]\n"
        "print(json.dumps(result))\n"
    )
    command = ["conda", "run", "-n", conda_env, "python", "-c", script, str(event_path)]
    try:
        result_proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        return {}
    if result_proc.returncode != 0:
        return {}
    try:
        loaded = json.loads(result_proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}
    return {key: [float(v) for v in values] for key, values in loaded.items() if isinstance(values, list)}


# ---------------------------------------------------------------------------
# Sampling + error selection
# ---------------------------------------------------------------------------

def _sample_metric_history(
    metric_values: dict[str, list[float]],
    max_points: int,
) -> list[dict]:
    max_count = max((len(values) for values in metric_values.values()), default=0)
    if max_count == 0:
        return []
    if max_points <= 1:
        indices = [max_count - 1]
    elif max_count <= max_points:
        indices = list(range(max_count))
    else:
        indices = sorted(
            {round(i * (max_count - 1) / (max_points - 1)) for i in range(max_points)}
        )

    history: list[dict] = []
    for index in indices:
        point = {"episode": index + 1}
        for metric_name, values in metric_values.items():
            if index < len(values):
                point[metric_name] = values[index]
        if len(point) > 1:
            history.append(point)
    return history


def _select_lines(lines: list[str], needles: tuple[str, ...], limit: int = 40) -> list[str]:
    matches = [line for line in lines if any(needle in line.lower() for needle in needles)]
    return matches[-limit:]
