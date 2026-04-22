"""Coder agent: generates reward-code candidates from the Planner artifact."""

from __future__ import annotations

from ..core import CodeArtifact, PlanArtifact, TaskBrief
from ..llm import QwenClient
from ..llm.parsers import parse_reward_code, require_code_block_terminator
from .logger import LiveLogger, null_logger
from .prompts import CODER_REVISION_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT, build_coder_prompt


class CoderAgent:
    """Generates the reward-code body inside ``_compute_generated_reward``."""

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

    def write_code(
        self,
        task_brief: TaskBrief,
        plan: PlanArtifact,
        *,
        previous_code: CodeArtifact | None = None,
        reviewer_feedback: str | None = None,
        critic_feedback: str | None = None,
    ) -> CodeArtifact:
        prompt = build_coder_prompt(
            plan,
            previous_code,
            reviewer_feedback=reviewer_feedback,
            critic_feedback=critic_feedback,
        )
        system_prompt = (
            CODER_REVISION_SYSTEM_PROMPT
            if reviewer_feedback is not None or critic_feedback is not None
            else CODER_SYSTEM_PROMPT
        )
        self.logger.log_full(prompt, agent="coder", kind="input")

        raw_response = self.client.generate_text(
            system_prompt,
            prompt,
            max_tokens=max(self.client.config.max_tokens, 2048),
            enable_thinking=self.enable_thinking,
        )
        self.logger.log_full(raw_response, agent="coder", kind="output")

        require_code_block_terminator(raw_response, "Coder")
        parsed = parse_reward_code(raw_response)
        self.logger.log_output(raw_response, agent="coder")

        return CodeArtifact(
            code=parsed.code,
            raw_response=parsed.raw_response,
            thinking=parsed.thinking,
            metadata={"agent": "coder"},
        )
