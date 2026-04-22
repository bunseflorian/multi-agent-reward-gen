"""Live conversation logging + final run persistence.

The previous ``run_logger`` module leaked global state via a module-level
``_LIVE_LOG_DIR``. Here ``LiveLogger`` is an explicit object that every agent
receives. Agents that do not need live logging receive :func:`null_logger`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..core import RunState
from ..llm.parsers import extract_logged_output


# ---------------------------------------------------------------------------
# LiveLogger
# ---------------------------------------------------------------------------

class LiveLogger:
    """Append streaming agent conversations to a run directory.

    Writes two log files:

    - ``full_conversation.md`` — complete prompt/response pairs
    - ``conversation.md`` — output-only transcript entries
    """

    def __init__(self, log_dir: Path | str | None):
        self._log_dir = Path(log_dir) if log_dir is not None else None

    @property
    def enabled(self) -> bool:
        return self._log_dir is not None

    def log_full(self, content: str, *, agent: str, kind: str) -> None:
        """Write a complete prompt or response into ``full_conversation.md``."""

        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat()
        with (self._log_dir / "full_conversation.md").open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n## {agent.title()} {kind.title()}\n\n")
            log_file.write(f"timestamp: `{timestamp}`\n\n")
            log_file.write("```text\n")
            log_file.write(content.rstrip() + "\n")
            log_file.write("```\n")

    def log_output(self, content: str, *, agent: str, extract: bool = True) -> None:
        """Append an output-only transcript entry into ``conversation.md``."""

        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat()
        logged = extract_logged_output(content) if extract else content.strip()
        with (self._log_dir / "conversation.md").open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n## {agent.title()} Output\n\n")
            log_file.write(f"timestamp: `{timestamp}`\n\n")
            log_file.write("```text\n")
            log_file.write(logged.rstrip() + "\n")
            log_file.write("```\n")


def null_logger() -> LiveLogger:
    """Return a logger that silently ignores all writes."""

    return LiveLogger(None)


# ---------------------------------------------------------------------------
# Final run persistence
# ---------------------------------------------------------------------------

def save_run_state(state: RunState, output_root: str | Path = "runs", *, run_dir: Path | None = None) -> Path:
    """Persist a multi-agent conversation and its artifacts to a run directory."""

    if run_dir is None:
        run_dir = Path(output_root) / run_dir_name(state)
        run_dir.mkdir(parents=True, exist_ok=False)

    _write_text(run_dir / "task_brief.md", _format_task_brief(state))
    _write_text(run_dir / "env_code_context.md", state.task_brief.env_code_context)

    if state.iterations:
        _write_text(run_dir / "plan.md", state.iterations[0].plan.text)
    if state.final_code is not None:
        _write_text(run_dir / "final_reward.py", state.final_code.code)
    return run_dir


def run_dir_name(state: RunState) -> str:
    """Return a timestamped, filesystem-safe run directory name."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = _safe_name(state.task_brief.task_id)
    return f"{timestamp}_{task_id}_{state.status.value}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")


def _format_task_brief(state: RunState) -> str:
    task = state.task_brief
    lines = [
        f"# Task: {task.task_id}",
        "",
        "## Objective",
        task.objective or "(derived from environment code)",
        "",
        "## Environment Source Path",
        task.env_source_path,
    ]
    lines += [
        "",
        "## RL Training",
        "skipped" if state.rl_training_skipped else "enabled",
        "",
        "## Final Status",
        state.status.value,
    ]
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
