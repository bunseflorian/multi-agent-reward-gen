"""Planner agent: turns a task brief into a natural-language reward plan."""

from __future__ import annotations

from ..core import PlanArtifact, TaskBrief
from ..llm import QwenClient
from ..llm.parsers import extract_final_answer, extract_plan_bullets, require_final_answer_terminator
from .logger import LiveLogger, null_logger
from .prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt


class PlannerAgent:
    """Creates the natural-language reward plan consumed by every later agent."""

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

    def plan(self, task_brief: TaskBrief) -> PlanArtifact:
        prompt = build_planner_prompt(task_brief)
        self.logger.log_full(prompt, agent="planner", kind="input")

        raw = self.client.generate_text(
            PLANNER_SYSTEM_PROMPT,
            prompt,
            max_tokens=max(self.client.config.max_tokens, 4096),
            enable_thinking=self.enable_thinking,
        )
        self.logger.log_full(raw, agent="planner", kind="output")

        require_final_answer_terminator(raw, "Planner")
        text = extract_final_answer(raw)
        self.logger.log_output(raw, agent="planner")
        return PlanArtifact(
            text=text,
            requirements=extract_plan_bullets(text),
            metadata={"agent": "planner"},
        )
