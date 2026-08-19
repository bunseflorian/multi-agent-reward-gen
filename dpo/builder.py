"""Build a single DPO preference pair from a completed RunState."""

from __future__ import annotations

import re

from ..core import RunState, RunStatus


def build_pair(state: RunState) -> dict | None:
    """Return {"prompt", "chosen", "rejected"} or None if no valid pair exists."""
    if state.status != RunStatus.ACCEPTED_RL:
        return None

    chosen = _trim_after_reward(state.final_code.code)

    last_revise = next(
        (r.code.code for r in reversed(state.iterations)
         if r.critique is not None and r.critique.should_retry),
        None,
    )
    last_reviewer_fail = next(
        (r.code.code for r in reversed(state.iterations)
         if r.code_review is not None and not r.code_review.passed),
        None,
    )
    rejected_raw = last_revise or last_reviewer_fail
    if rejected_raw is None:
        return None

    rejected = _trim_after_reward(rejected_raw)

    if rejected == chosen:
        return None

    prompt = (
        f"<env_code>\n{state.task_brief.env_code_context}"
        f"\n\n<user_prompt>\n{state.task_brief.objective}"
    )
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _trim_after_reward(code: str) -> str:
    """Keep everything up to and including the last 'reward = ...' assignment."""
    lines = code.splitlines()
    last_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*reward\s*=(?!=)", line):
            last_idx = i
    if last_idx is None:
        return code
    return "\n".join(lines[: last_idx + 1])
