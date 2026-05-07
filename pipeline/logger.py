"""Run artifact recording backed by one JSONL event log."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..core import RunState
from ..llm.parsers import extract_final_answer, extract_logged_output


# ---------------------------------------------------------------------------
# RunRecorder
# ---------------------------------------------------------------------------

class RunRecorder:
    """Record run events and derive human-readable artifacts from them."""

    def __init__(self, log_dir: Path | str | None):
        self._log_dir = Path(log_dir) if log_dir is not None else None

    @property
    def enabled(self) -> bool:
        return self._log_dir is not None

    @property
    def run_dir(self) -> Path | None:
        return self._log_dir

    def move_to(self, run_dir: Path | str) -> None:
        """Move the current run directory and continue recording there."""

        new_dir = Path(run_dir)
        if self._log_dir is not None and self._log_dir.exists() and self._log_dir != new_dir:
            if new_dir.exists():
                raise FileExistsError(new_dir)
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            self._log_dir.rename(new_dir)
        else:
            new_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = new_dir

    def record_initial_artifacts(self, state: RunState) -> None:
        """Write stable audit/debug artifacts for the run."""

        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        _write_text(self._log_dir / "task_brief.md", _format_task_brief(state))
        _write_text(self._log_dir / "env_code_context.md", state.task_brief.env_code_context)

    def event(self, event_type: str, **payload: Any) -> None:
        """Append a structured event to ``events.jsonl``."""

        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        with (self._log_dir / "events.jsonl").open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True) + "\n")

    def log_full(self, content: str, *, agent: str, kind: str) -> None:
        """Record a complete prompt or response."""

        self.event(
            "agent_prompt" if kind == "input" else "agent_output",
            agent=agent,
            kind=kind,
            content=content,
        )
        self.render_markdown_logs()

    def log_output(self, content: str, *, agent: str, extract: bool = True) -> None:
        """Record an output-only transcript entry.

        Thinking blocks are always stripped. ``extract=True`` additionally
        isolates the last fenced Python block (used for the coder); other
        agents pass ``extract=False`` to keep their full text-report output.
        """

        logged = extract_logged_output(content) if extract else extract_final_answer(content)
        self.event("conversation_entry", agent=agent, content=logged)
        self.render_markdown_logs()

    def save_run_state(self, state: RunState) -> Path | None:
        """Persist final artifacts and a complete structured ``run_state.json``."""

        if self._log_dir is None:
            return None
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.record_initial_artifacts(state)
        plan = state.plan or (state.iterations[0].plan if state.iterations else None)
        if plan is not None:
            _write_text(self._log_dir / "plan.md", plan.text)
        if state.final_code is not None:
            _write_text(self._log_dir / "final_reward.py", state.final_code.code)
        self.event("run_state", state=state)
        _write_json(self._log_dir / "run_state.json", state)
        self.render_markdown_logs()
        return self._log_dir

    def render_markdown_logs(self) -> None:
        """Render ``conversation.md`` and ``full_conversation.md`` from events."""

        if self._log_dir is None:
            return
        events = _read_events(self._log_dir / "events.jsonl")
        _write_text(self._log_dir / "conversation.md", _render_conversation(events))
        _write_text(self._log_dir / "full_conversation.md", _render_full_conversation(events))


def null_logger() -> RunRecorder:
    """Return a recorder that silently ignores all writes."""

    return RunRecorder(None)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            events.append(loaded)
    return events


def _render_conversation(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        if event.get("event") != "conversation_entry":
            continue
        parts.extend(_markdown_block(f"{event.get('agent', 'agent').title()} Output", event))
    return "\n".join(parts).strip() + "\n" if parts else ""


def _render_full_conversation(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        event_type = event.get("event")
        if event_type not in {"agent_prompt", "agent_output"}:
            continue
        label = "Input" if event_type == "agent_prompt" else "Output"
        parts.extend(_markdown_block(f"{event.get('agent', 'agent').title()} {label}", event))
    return "\n".join(parts).strip() + "\n" if parts else ""


def _markdown_block(title: str, event: dict[str, Any]) -> list[str]:
    return [
        f"## {title}",
        "",
        f"timestamp: `{event.get('timestamp', '')}`",
        "",
        "```text",
        str(event.get("content", "")).rstrip(),
        "```",
        "",
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        data = asdict(value)
        data.pop("recorder", None)
        return _jsonable(data)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
