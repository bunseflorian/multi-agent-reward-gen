"""Compact text rendering of a completed ``RunState`` for terminal use."""

from __future__ import annotations

from ..core import RunState


def print_state(state: RunState) -> None:
    """Print a task summary, iteration trace, plan, errors, and final code."""

    print(f"Task: {state.task_brief.task_id}")
    print(f"Status: {state.status.value}")
    print(f"Iterations: {len(state.iterations)}")
    print("")

    for record in state.iterations:
        print(f"Iteration {record.iteration}")
        if record.code_review is not None:
            print(f"  code_review_passed: {record.code_review.passed}")
        if record.ppo_report is not None:
            print(f"  rl_status: {record.ppo_report.status}")
            if record.ppo_report.log_path:
                print(f"  ppo_log: {record.ppo_report.log_path}")
        if record.critique:
            print("  critique:")
            print(_indent(record.critique.feedback, "    "))
        print("")

    if state.iterations:
        print("Plan:")
        print(_indent(state.iterations[0].plan.text, "  "))
        print("")

    if state.final_report is not None and state.final_report.errors:
        print("Errors:")
        for error in state.final_report.errors:
            print(f"  - {error}")
        print("")

    if state.final_code:
        print("Final reward code:")
        print("```python")
        print(state.final_code.code)
        print("```")


def _indent(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())
