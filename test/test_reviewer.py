"""Run the Reviewer against a realistic (plan, code, static-report) triple."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import call_llm, load_prompt, print_banner  # noqa: E402


PLAN_TEXT = """Extracted behavior terms:
- term: Minimize distance to the target position.

Primary objective:
- objective: Minimize distance to the target position.
  task type: event
  resolved from tensors:
    - self._success
    - self._dist_to_target
  metrics to log:
    - final_success: Task completion flag at episode end.
    - mean_dist_to_target: Average distance to target over episode.
  metric trends:
    - final_success: should approach 1 over training
    - mean_dist_to_target: should approach 0.0 over training

Constraints:
- (None)

Signal contract:
- self._success [shape: (N,)]: task completion flag; higher is better; event; use: success bonus; derived from: self._close_step_count.
- self._dist_to_target [shape: (N,)]: distance to target; lower is better; dense; use: error-to-bonus; derived from: self._robot.data.root_pos_w, self._desired_pos_w.
"""

GENERATED_CODE = """# Raw signals
dist_to_target = self._dist_to_target
is_success = self._success

# Reward components
dist_bonus = 1.5 * torch.exp(-5.0 * dist_to_target / 3.0)
success_bonus = 1.0 * is_success.float()

# Final reward
reward = dist_bonus + success_bonus

# Reward-term sums
self._episode_final_values.setdefault("final_reward_term_dist_bonus", torch.zeros_like(dist_bonus))
self._episode_final_values["final_reward_term_dist_bonus"] += dist_bonus
self._episode_final_values.setdefault("final_reward_term_success_bonus", torch.zeros_like(success_bonus))
self._episode_final_values["final_reward_term_success_bonus"] += success_bonus

# Metrics
self._episode_sums.setdefault("mean_dist_to_target", torch.zeros_like(dist_to_target))
self._episode_sums["mean_dist_to_target"] += dist_to_target
self._episode_final_values["final_success"] = is_success.float()
"""

STATIC_REPORT_PASSED = True
STATIC_ERRORS: list[str] = []
STATIC_WARNINGS: list[str] = []


def main() -> None:
    system_prompt = load_prompt("REVIEWER_SYSTEM_PROMPT")

    sections = [
        "Plan:",
        PLAN_TEXT,
        "",
        "Generated reward code:",
        "```python",
        GENERATED_CODE,
        "```",
        "",
        f"Static check result: {'PASSED' if STATIC_REPORT_PASSED else 'FAILED'}",
    ]
    if STATIC_ERRORS:
        sections.append("Static check errors:")
        sections.extend(f"  - {e}" for e in STATIC_ERRORS)
    if STATIC_WARNINGS:
        sections.append("Static check warnings:")
        sections.extend(f"  - {w}" for w in STATIC_WARNINGS)
    if not STATIC_REPORT_PASSED:
        sections.append(
            "The static checker already found issues. "
            "Confirm them in your FEEDBACK and set REVIEW: FAIL."
        )

    output = call_llm(system_prompt, "\n".join(sections), max_tokens=1024)
    print_banner("REVIEWER OUTPUT")
    print(output)


if __name__ == "__main__":
    main()
