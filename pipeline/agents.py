"""Functional agent implementations for reward-code generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..core import CodeArtifact, CodeReviewArtifact, CritiqueArtifact, PlanArtifact, PPOReport, TaskBrief, TestReport
from ..llm import QwenClient
from ..llm.parsers import (
    critic_requests_revision,
    extract_final_answer,
    extract_logged_output,
    extract_plan_bullets,
    extract_plan_tensor_contract,
    extract_review_feedback,
    parse_reward_code,
    require_code_block_terminator,
    review_passed,
)
from .formatting import format_metric_history
from .logger import RunRecorder, null_logger
from .prompts import (
    CODER_FEEDBACK_PROMPT,
    CODER_INITIAL_PROMPT,
    CODER_RULES_PROMPT,
    CODER_USER,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER,
)
from .static_check import static_check as _static_check_direct


_STATIC_CHECK_SCRIPT = str(Path(__file__).parent / "static_check.py")


# ── shared LLM call ───────────────────────────────────────────────────────────

def _call(
    client: QwenClient,
    system: str,
    user: str,
    parser,
    *,
    agent: str,
    enable_thinking: bool = False,
    logger: RunRecorder | None = None,
    max_tokens: int = 6500,
) -> Any:
    if logger is None:
        logger = null_logger()
    logger.write(user, agent=agent, kind="input")
    raw = client.generate_text(
        system, user,
        max_tokens=max(max_tokens, getattr(client.config, "max_tokens", max_tokens)),
        enable_thinking=enable_thinking,
    )
    artifact, logged = parser(raw)
    logger.write(logged, agent=agent, kind="output", raw=raw)
    return artifact


# ── static check ─────────────────────────────────────────────────────────────

def _run_static_check(code: str, allowed_tensors: list[str] | None) -> TestReport:
    from ..configs import config_get
    try:
        conda_env = config_get("paths.isaaclab_conda_env")
    except (ValueError, KeyError):
        conda_env = None

    if not conda_env:
        return _static_check_direct(code, allowed_tensors)

    payload = json.dumps({"code": code, "allowed_tensors": allowed_tensors})
    try:
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", conda_env, "python", _STATIC_CHECK_SCRIPT],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return TestReport(
                passed=False,
                errors=[f"static_check subprocess failed (rc={result.returncode}): {result.stderr.strip()[:300]}"],
            )
        data = json.loads(result.stdout)
        return TestReport(
            passed=data["passed"],
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            metadata=data.get("metadata", {}),
        )
    except Exception as exc:
        return TestReport(passed=False, errors=[f"static_check subprocess error: {exc}"])


# ── planner ───────────────────────────────────────────────────────────────────

def planner(
    client: QwenClient,
    task_brief: TaskBrief,
    *,
    enable_thinking: bool = False,
    logger: RunRecorder | None = None,
) -> PlanArtifact:
    user = PLANNER_USER.format(
        task_id=task_brief.task_id,
        objective=task_brief.objective or "(to be derived from environment code)",
        env_source_path=task_brief.env_source_path,
        env_code_context=task_brief.env_code_context,
    )

    def parse(raw: str) -> tuple[PlanArtifact, str]:
        text = extract_final_answer(raw)
        return PlanArtifact(text=text, requirements=extract_plan_bullets(text), metadata={"agent": "planner"}), text

    return _call(client, PLANNER_SYSTEM_PROMPT, user, parse, agent="planner", enable_thinking=enable_thinking, logger=logger)


# ── coder ─────────────────────────────────────────────────────────────────────

def coder(
    client: QwenClient,
    plan: PlanArtifact,
    previous_code: CodeArtifact | None,
    reviewer_feedback: str | None,
    critic_feedback: str | None,
    *,
    enable_thinking: bool = False,
    logger: RunRecorder | None = None,
) -> CodeArtifact:
    system = (CODER_FEEDBACK_PROMPT if (reviewer_feedback or critic_feedback) else CODER_INITIAL_PROMPT) + "\n\n" + CODER_RULES_PROMPT

    parts: list[str] = []
    if reviewer_feedback:
        parts.append(f"Feedback from the code reviewer:\n{reviewer_feedback}")
        parts.append("Treat this as a repair delta. Preserve the Planner artifact below as the single source of truth.")
        if previous_code:
            parts.append(f"\nPrevious reward code implementation:\n```python\n{previous_code.code}\n```")
    if critic_feedback:
        parts.append(f"Context from RL Critic agent (PPO metric outcome; reward-shaping revision):\n{critic_feedback}")
        parts.append("Treat this as a repair delta. Preserve the Planner artifact below as the single source of truth.")
        if previous_code:
            parts.append(f"\nPrevious reward code implementation:\n```python\n{previous_code.code}\n```")
    user = ("\n".join(parts) + "\n" if parts else "") + CODER_USER.format(plan_text=plan.text)

    def parse(raw: str) -> tuple[CodeArtifact, str]:
        require_code_block_terminator(raw, "Coder")
        parsed = parse_reward_code(raw)
        return CodeArtifact(
            code=parsed.code,
            raw_response=parsed.raw_response,
            thinking=parsed.thinking,
            metadata={"agent": "coder"},
        ), extract_logged_output(raw)

    return _call(client, system, user, parse, agent="coder", enable_thinking=enable_thinking, logger=logger)


# ── reviewer ──────────────────────────────────────────────────────────────────

def reviewer(
    client: QwenClient,
    plan: PlanArtifact,
    code: CodeArtifact,
    *,
    enable_thinking: bool = False,
    logger: RunRecorder | None = None,
) -> CodeReviewArtifact:
    static_report = _run_static_check(code.code, extract_plan_tensor_contract(plan.text))

    static_parts: list[str] = []
    if static_report.errors:
        static_parts.append("Static check errors:\n" + "\n".join(f"  - {e}" for e in static_report.errors))
    if static_report.warnings:
        static_parts.append("Static check warnings:\n" + "\n".join(f"  - {w}" for w in static_report.warnings))
    if not static_report.passed:
        static_parts.append("The static checker already found issues. Confirm them in your FEEDBACK and set REVIEW: FAIL.")

    user = REVIEWER_USER.format(
        plan_text=plan.text,
        code=code.code,
        static_result="PASSED" if static_report.passed else "FAILED",
        static_errors="\n".join(static_parts),
    )

    def parse(raw: str) -> tuple[CodeReviewArtifact, str]:
        result = extract_final_answer(raw)
        passed = review_passed(result) and static_report.passed
        feedback = extract_review_feedback(result)
        return CodeReviewArtifact(
            passed=passed,
            feedback=feedback,
            raw_response=result,
            errors=[] if passed else [feedback or "Reviewer rejected the reward-code candidate."],
            metadata={
                "agent": "reviewer",
                "static_passed": static_report.passed,
                "static_errors": static_report.errors,
                "static_warnings": static_report.warnings,
            },
        ), result

    return _call(client, REVIEWER_SYSTEM_PROMPT, user, parse, agent="reviewer", enable_thinking=enable_thinking, logger=logger)


# ── critic ────────────────────────────────────────────────────────────────────

def critic(
    client: QwenClient,
    plan: PlanArtifact,
    code: CodeArtifact,
    ppo_report: PPOReport,
    *,
    enable_thinking: bool = False,
    logger: RunRecorder | None = None,
) -> CritiqueArtifact:
    raw_errors = list(ppo_report.errors)
    if isinstance(ppo_report.summary, dict):
        summary_errors = ppo_report.summary.get("errors", [])
        if isinstance(summary_errors, list):
            raw_errors.extend(summary_errors)
    seen: set[str] = set()
    unique = [e.strip() for e in raw_errors if isinstance(e, str) and e.strip() and not (seen := seen | {e.strip()}) - seen]
    runtime_errors = "\n".join(f"- {e}" for e in unique[:8]) if unique else "- none"

    summary = ppo_report.summary if isinstance(ppo_report.summary, dict) else {}
    critic_metric_history = summary.get("critic_metric_history")
    metric_history = (
        critic_metric_history
        if isinstance(critic_metric_history, str) and critic_metric_history.strip()
        else format_metric_history(summary.get("metric_history", []))
    )

    user = CRITIC_USER.format(
        plan_text=plan.text,
        code=code.code,
        status=ppo_report.status,
        passed=ppo_report.passed,
        timed_out=ppo_report.timed_out,
        exit_code=ppo_report.exit_code,
        runtime_errors=runtime_errors,
        metric_history=metric_history,
    )

    def parse(raw: str) -> tuple[CritiqueArtifact, str]:
        result = extract_final_answer(raw)
        return CritiqueArtifact(
            feedback=result,
            should_retry=critic_requests_revision(result, ppo_report.passed),
            metadata={"agent": "critic", "source": "ppo_training"},
        ), result

    return _call(client, CRITIC_SYSTEM_PROMPT, user, parse, agent="critic", enable_thinking=enable_thinking, logger=logger, max_tokens=4096)
