"""Reviewer agent: runs the static gate, then asks the LLM to confirm compliance."""

from __future__ import annotations

from ..core import CodeArtifact, CodeReviewArtifact, PlanArtifact
from ..llm import QwenClient
from ..llm.parsers import (
    extract_final_answer,
    extract_plan_tensor_contract,
    extract_review_feedback,
    review_passed,
)
from .logger import RunRecorder, null_logger
from .prompts import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from .static_check import static_check


class ReviewerAgent:
    """Wraps the deterministic static check with an LLM compliance review."""

    def __init__(
        self,
        client: QwenClient,
        *,
        enable_thinking: bool = False,
        logger: RunRecorder | None = None,
    ):
        self.client = client
        self.enable_thinking = enable_thinking
        self.logger = logger or null_logger()

    def review(
        self,
        plan: PlanArtifact,
        code_artifact: CodeArtifact,
    ) -> CodeReviewArtifact:
        static_report = static_check(
            code_artifact.code,
            allowed_tensors=extract_plan_tensor_contract(plan.text),
        )

        prompt = build_reviewer_prompt(plan, code_artifact, static_report=static_report)
        self.logger.log_full(prompt, agent="reviewer", kind="input")

        agent_output = self.client.generate_text(
            REVIEWER_SYSTEM_PROMPT,
            prompt,
            max_tokens=6500,
            enable_thinking=self.enable_thinking,
        )
        self.logger.log_full(agent_output, agent="reviewer", kind="output")

        raw_response = extract_final_answer(agent_output)
        passed = review_passed(raw_response) and static_report.passed
        feedback = extract_review_feedback(raw_response)
        self.logger.log_output(agent_output, agent="reviewer", extract=False)

        errors = [] if passed else [feedback or "Reviewer rejected the reward-code candidate."]
        return CodeReviewArtifact(
            passed=passed,
            feedback=feedback,
            raw_response=raw_response,
            errors=errors,
            metadata={
                "agent": "reviewer",
                "static_passed": static_report.passed,
                "static_errors": static_report.errors,
                "static_warnings": static_report.warnings,
            },
        )
