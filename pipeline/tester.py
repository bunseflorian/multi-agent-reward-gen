"""Tester agent: optional PPO validation wrapper."""

from __future__ import annotations

from ..core import CodeArtifact, PPOConfig, PPOReport, TaskBrief
from ..ppo import run_ppo_validation, skipped_ppo_report


class TesterAgent:
    """Runs PPO validation on a candidate reward or returns a skipped report."""

    def test(
        self,
        task_brief: TaskBrief,
        code_artifact: CodeArtifact,
        ppo_config: PPOConfig,
        run_dir: str | None = None,
    ) -> PPOReport:
        if not ppo_config.run_rl_training:
            return skipped_ppo_report()
        if run_dir is None:
            raise ValueError("run_dir is required when RL training is enabled.")
        if ppo_config.hooks.pre_training_callback is not None:
            ppo_config.hooks.pre_training_callback()
        try:
            return run_ppo_validation(task_brief, code_artifact.code, run_dir, ppo_config)
        finally:
            if ppo_config.hooks.post_training_callback is not None:
                ppo_config.hooks.post_training_callback()
