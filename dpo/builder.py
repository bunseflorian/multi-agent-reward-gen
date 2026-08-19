"""Build a single DPO preference pair from a completed RunState."""

from __future__ import annotations

from ..core import RunState, RunStatus


def build_pair(state: RunState) -> dict | None:
    """Return {"prompt", "chosen", "rejected"} or None if no valid pair exists."""
    if state.status != RunStatus.ACCEPTED_RL:
        return None

    chosen = state.final_code.code

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
    rejected = last_revise or last_reviewer_fail
    if rejected is None:
        return None
    if rejected == chosen:
        return None

    prompt = (
        f"<env_code>\n{state.task_brief.env_code_context}"
        f"\n\n<user_prompt>\n{state.task_brief.objective}"
    )
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
