"""Run the Critic against the on-track quadcopter trajectory.

Expected decision: ACCEPT. The acceptance metric ``final_success`` reaches 1.0
in the last quarter of the sampled episodes and the dense diagnostic
``mean_dist_to_target`` has converged from ~1.7 m to ~0.2 m.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import call_llm, load_prompt, print_banner  # noqa: E402


USER_PROMPT = """Plan:
Extracted behavior terms:
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
    - self._dist_to_target: Euclidean distance between robot root and target (shape: (N,)). Updated in `_get_rewards` as `torch.norm(self._robot.data.root_pos_w - self._desired_pos_w)`.
  metrics to log:
    - final_success: Task completion flag at episode end.
    - mean_dist_to_target: Average distance to target over episode.
  metric trends:
    - final_success: should approach 1.0 over training
    - mean_dist_to_target: should approach 0.0 over training

Constraints:

Signal contract:
- self._success [shape: (N,)]: task completion flag; higher is better; event; use: success bonus; derived from: self._close_step_count.
- self._close_step_count [shape: (N,)]: consecutive close-target counter; higher is better; stepwise; use: metric; derived from: self._dist_to_target.
- self._dist_to_target [shape: (N,)]: distance to target; lower is better; dense; use: error-to-bonus; derived from: self._robot.data.root_pos_w, self._desired_pos_w.
- self._desired_pos_w [shape: (N, 3)]: target position; target is better; diagnostic; use: metric; derived from: none.
- self._robot.data.root_pos_w [shape: (N, 3)]: robot root position; target is better; diagnostic; use: metric; derived from: none.

Task scale hints:
- target sampling range: x,y: ~0-4 m, z: ~0.5-1.5 m (from _desired_pos_w reset range)
- relevant error scale: dist_to_target: ~0-3 m
- episode horizon: 10.0 s | physics dt: 0.01 s | decimation: 2 | control frequency: 50 Hz

Generated code:
# Raw signals
dist_to_target = self._dist_to_target
is_success = self._success

# Primary objective bonus: exponential decay on distance
dist_bonus_amp = 1.5
sharpness = 5.0
scale_dist = 3.0
dist_bonus = dist_bonus_amp * torch.exp(-sharpness * dist_to_target / scale_dist)

# Success bonus: binary completion reward
success_bonus_amp = 1.0
success_bonus = success_bonus_amp * is_success.float()

# Final reward
reward = dist_bonus + success_bonus

# Reward-term sums
self._episode_final_values.setdefault("final_reward_term_dist_bonus", torch.zeros_like(dist_bonus))
self._episode_final_values["final_reward_term_dist_bonus"] += dist_bonus
self._episode_final_values.setdefault("final_reward_term_success_bonus", torch.zeros_like(success_bonus))
self._episode_final_values["final_reward_term_success_bonus"] += success_bonus

# Metric stores
self._episode_sums.setdefault("mean_dist_to_target", torch.zeros_like(dist_to_target))
self._episode_sums["mean_dist_to_target"] += dist_to_target
self._episode_final_values["final_success"] = is_success.float()

PPO report summary:
status=passed
passed=True
timed_out=False
exit_code=0

PPO runtime errors:
- none

Sampled PPO episode metrics passed to the Critic:
final_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16, 0.307692, 0.75, 0.466667, 0, 0.6, 0.4, 0.571429, 0.6, 0.666667, 0.625, 0.583333, 0.785714, 0.75, 1, 0.545455, 0.5, 0.875, 1, 0.846154, 1, 1, 1, 1, 0.888889, 1]
mean_dist_to_target: [1.68433, 1.68387, 1.68544, 1.66317, 1.66612, 1.60137, 1.49098, 1.38565, 1.16336, 0.997812, 0.85585, 0.6574, 0.415435, 0.368686, 0.29385, 0.260285, 0.223902, 0.244374, 0.241283, 0.213458, 0.250061, 0.192738, 0.24617, 0.214088, 0.209356, 0.249453, 0.201873, 0.218167, 0.198312, 0.232566, 0.261972, 0.208528, 0.199592, 0.259193, 0.190551, 0.193871, 0.211601, 0.199976, 0.203871, 0.192667]
"""


def main() -> None:
    system_prompt = load_prompt("CRITIC_SYSTEM_PROMPT")
    output = call_llm(system_prompt, USER_PROMPT, max_tokens=2048)
    print_banner("CRITIC OUTPUT")
    print(output)


if __name__ == "__main__":
    main()
