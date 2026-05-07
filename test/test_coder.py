"""Run the Coder against a realistic Planner artifact (no reviewer/critic feedback)."""

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
    - self._close_step_count
    - self._dist_to_target
  tensors and semantics:
    - self._success: Binary success flag (shape: (N,)) indicating task completion. Updated in `_get_rewards` when `self._close_step_count >= self._success_steps_required`.
    - self._close_step_count: Counter tracking consecutive steps where distance is below threshold. Updated in `_get_rewards` incrementally.
    - self._dist_to_target: Euclidean distance between robot root and target (shape: (N,)). Updated in `_get_rewards` as `torch.norm(self._robot.data.root_pos_w - self._desired_pos_w, dim=-1)`.
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
- self._close_step_count [shape: (N,)]: consecutive close-target counter; higher is better; stepwise; use: metric; derived from: self._dist_to_target.
- self._dist_to_target [shape: (N,)]: distance to target; lower is better; dense; use: error-to-bonus; derived from: self._robot.data.root_pos_w, self._desired_pos_w.
- self._desired_pos_w [shape: (N, 3)]: target position; target is better; diagnostic; use: metric; derived from: none.
- self._robot.data.root_pos_w [shape: (N, 3)]: robot root position; target is better; diagnostic; use: metric; derived from: none.

Task scale hints:
- target sampling range: x,y ∈ [-2, 2] m, z ∈ [0.5, 1.5] m (from _desired_pos_w reset range)
- relevant error scale: dist_to_target: ~0–3 m
- episode horizon: 10.0 s | physics dt: 0.01 s | decimation: 2 | control frequency: 50 Hz
"""


def main() -> None:
    system_prompt = "\n\n".join(
        [load_prompt("CODER_INITIAL_PROMPT"), load_prompt("CODER_RULES_PROMPT")]
    )
    user_prompt = "\n".join(
        [
            "Context from Planner agent (reward design intent):",
            PLAN_TEXT,
        ]
    )
    output = call_llm(system_prompt, user_prompt, max_tokens=2048)
    print_banner("CODER OUTPUT")
    print(output)


if __name__ == "__main__":
    main()
