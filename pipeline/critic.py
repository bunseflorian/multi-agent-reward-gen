"""Critic agent: reads PPO metrics and returns ACCEPT or REVISE."""

from __future__ import annotations

from ..core import CodeArtifact, CritiqueArtifact, PlanArtifact, PPOReport, TaskBrief
from ..llm import QwenClient
from ..llm.parsers import critic_requests_revision, extract_final_answer, require_final_answer_terminator
from .logger import LiveLogger, null_logger
from .prompts import CRITIC_SYSTEM_PROMPT, build_critic_prompt


class CriticAgent:
    """Evaluates PPO outcomes and decides whether the reward should be revised."""

    def __init__(
        self,
        client: QwenClient,
        *,
        enable_thinking: bool = False,
        logger: LiveLogger | None = None,
    ):
        self.client = client
        self.enable_thinking = enable_thinking
        self.logger = logger or null_logger()

    def critique(
        self,
        task_brief: TaskBrief,
        plan: PlanArtifact,
        code: CodeArtifact,
        ppo_report: PPOReport | None = None,
    ) -> CritiqueArtifact:
        if ppo_report is None or ppo_report.status == "skipped":
            return CritiqueArtifact(
                feedback="",
                should_retry=False,
                metadata={"agent": "critic", "source": "ppo_not_run", "skipped": True},
            )

        prompt = build_critic_prompt(task_brief, plan, code, ppo_report)
        self.logger.log_full(prompt, agent="critic", kind="input")

        agent_output = self.client.generate_text(
            CRITIC_SYSTEM_PROMPT,
            prompt,
            max_tokens=max(self.client.config.max_tokens, 2048),
            enable_thinking=self.enable_thinking,
        )
        self.logger.log_full(agent_output, agent="critic", kind="output")

        require_final_answer_terminator(agent_output, "Critic")
        feedback = extract_final_answer(agent_output)
        self.logger.log_output(agent_output, agent="critic")

        return CritiqueArtifact(
            feedback=feedback,
            should_retry=critic_requests_revision(feedback, ppo_report.passed),
            metadata={"agent": "critic", "source": "ppo_training"},
        )
