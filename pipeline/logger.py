"""Conversation logger — appends agent turns to a single conversation.md."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..core import RunState


class RunRecorder:
    def __init__(self, log_dir: Path | str | None):
        self._log_dir = Path(log_dir) if log_dir is not None else None

    def move_to(self, run_dir: Path | str) -> None:
        new_dir = Path(run_dir)
        if self._log_dir is not None and self._log_dir.exists() and self._log_dir != new_dir:
            if new_dir.exists():
                raise FileExistsError(new_dir)
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            self._log_dir.rename(new_dir)
        else:
            new_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = new_dir

    def write(self, content: str, *, agent: str, kind: str, raw: str | None = None) -> None:
        """Append one agent turn to conversation.md and conversation_full.md."""
        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"## {agent.title()} — {kind.title()}\n*{timestamp}*\n"
        block = f"{header}\n```\n{content.strip()}\n```\n\n---\n\n"

        # conversation.md: parsed outputs only
        if kind == "output":
            with (self._log_dir / "conversation.md").open("a", encoding="utf-8") as f:
                f.write(block)

        # conversation_full.md: all inputs + raw LLM outputs
        full_content = raw if (kind == "output" and raw is not None) else content
        full_block = f"{header}\n```\n{full_content.strip()}\n```\n\n---\n\n"
        with (self._log_dir / "conversation_full.md").open("a", encoding="utf-8") as f:
            f.write(full_block)


def null_logger() -> RunRecorder:
    return RunRecorder(None)


def run_dir_name(state: RunState) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in state.task_brief.task_id).strip("_")
    return f"{timestamp}_{task_id}_{state.status.value}"
