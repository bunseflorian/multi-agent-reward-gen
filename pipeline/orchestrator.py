"""LangGraph-based orchestrator: Planner → Coder → Reviewer → Tester → Critic."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..core import (
    CodeArtifact,
    CodeReviewArtifact,
    CritiqueArtifact,
    IterationRecord,
    PlanArtifact,
    PPOConfig,
    PPOReport,
    RunStatus,
    TaskBrief,
    TestReport,
)
from ..llm import QwenApiError, QwenClient
from . import agents
from .formatting import format_ppo_live_log
from .logger import RunRecorder
from .tester import TesterAgent


class AgentState(TypedDict):
    # ── read-only config (set once at graph entry) ──
    task_brief: TaskBrief
    ppo_config: PPOConfig
    max_iterations: int
    max_rl_iterations: int
    max_format_retries: int
    stop_after_plan: bool
    run_dir: str | None
    agent_reasoning: dict[str, bool]
    enable_thinking: dict[str, bool]
    # ── agent outputs ──
    plan: PlanArtifact | None
    code: CodeArtifact | None
    code_review: CodeReviewArtifact | None
    ppo_report: PPOReport | None
    critique: CritiqueArtifact | None
    # ── loop control ──
    reviewer_feedback: str | None
    critic_feedback: str | None
    iteration: int
    rl_runs_started: int
    reviewer_blocked_rl_attempts: int
    format_retries: int
    # ── final result ──
    status: str                   # RunStatus.value string
    final_code: CodeArtifact | None
    final_report: TestReport | None
    final_ppo_report: PPOReport | None
    iterations: list              # list[IterationRecord]
    recorder: Any                 # RunRecorder, not serialized by LangGraph


# ── helpers ───────────────────────────────────────────────────────────────────

def _budget_exhausted(state: AgentState) -> bool:
    if state["ppo_config"].run_rl_training:
        return state["rl_runs_started"] >= state["max_rl_iterations"]
    return state["iteration"] >= state["max_iterations"]


def _make_test_report(code_review: CodeReviewArtifact) -> TestReport:
    return TestReport(
        passed=code_review.passed,
        errors=code_review.errors,
        warnings=code_review.warnings,
        metadata={"agent": "reviewer", "stage": "static_llm_validation"},
    )


def _reviewer_blocked_report(
    base_report: TestReport,
    max_iterations: int,
    reviewer_blocked_rl_attempts: int,
) -> TestReport:
    return TestReport(
        passed=False,
        errors=[
            f"RL training could not start because the reviewer rejected {max_iterations} generated candidates.",
            *base_report.errors,
        ],
        warnings=base_report.warnings,
        metadata={**base_report.metadata, "reviewer_blocked_rl_attempts": reviewer_blocked_rl_attempts},
    )


# ── graph factory ─────────────────────────────────────────────────────────────

def build_graph(client: QwenClient, *, recorder: RunRecorder):  # noqa: ARG001
    """Build and compile the reward-code generation StateGraph."""

    def planner_node(state: AgentState) -> Command:
        if state["plan"] is not None:
            return Command(goto="coder")
        try:
            plan = agents.planner(
                client, state["task_brief"],
                enable_thinking=state["enable_thinking"].get("planner", False),
                logger=state["recorder"],
            )
            return Command(
                update={"plan": plan, "status": RunStatus.PLANNED.value},
                goto=END if state["stop_after_plan"] else "coder",
            )
        except QwenApiError:
            raise
        except Exception as exc:
            return Command(
                update={
                    "status": RunStatus.FAILED_PLANNING.value,
                    "final_report": TestReport(
                        passed=False,
                        errors=[f"Planner failed: {type(exc).__name__}: {exc}"],
                        metadata={"agent": "planner", "stage": "planning"},
                    ),
                },
                goto=END,
            )

    def coder_node(state: AgentState) -> Command:
        iteration = state["iteration"] + 1
        last_exc: ValueError | None = None
        for _ in range(max(1, state["max_format_retries"])):
            try:
                code = agents.coder(
                    client,
                    state["plan"],
                    state["code"],
                    state["reviewer_feedback"],
                    state["critic_feedback"],
                    enable_thinking=state["enable_thinking"].get("coder", False),
                    logger=state["recorder"],
                )
                return Command(
                    update={"code": code, "iteration": iteration, "status": RunStatus.CODE_GENERATED.value, "format_retries": 0},
                    goto="reviewer",
                )
            except ValueError as exc:
                last_exc = exc
        return Command(
            update={
                "iteration": iteration,
                "status": RunStatus.FAILED_FORMAT.value,
                "final_report": TestReport(passed=False, errors=[f"Output format error: {last_exc}"], metadata={"stage": "format_check"}),
            },
            goto=END,
        )

    def reviewer_node(state: AgentState) -> Command:
        code_review = agents.reviewer(
            client,
            state["plan"],
            state["code"],
            enable_thinking=state["enable_thinking"].get("reviewer", False),
            logger=state["recorder"],
        )
        if code_review.passed:
            return Command(
                update={
                    "code_review": code_review,
                    "status": RunStatus.VALIDATED_STATIC.value,
                    "reviewer_feedback": None,
                    "reviewer_blocked_rl_attempts": 0,
                },
                goto="tester",
            )
        blocked = state["reviewer_blocked_rl_attempts"] + 1
        update = {
            "code_review": code_review,
            "status": RunStatus.FAILED_VALIDATION.value,
            "reviewer_feedback": code_review.feedback,
            "reviewer_blocked_rl_attempts": blocked,
        }
        if state["ppo_config"].run_rl_training and blocked >= state["max_iterations"]:
            return Command(update=update, goto="finalizer")
        if _budget_exhausted(state):
            return Command(update=update, goto="finalizer")
        return Command(update=update, goto="coder")

    def tester_node(state: AgentState) -> Command:
        ppo_config = state["ppo_config"]
        if ppo_config.run_rl_training:
            rl_runs_started = state["rl_runs_started"] + 1
            ppo_run_dir = (
                str(Path(state["run_dir"]) / f"rl_run_{rl_runs_started:02d}")
                if state["run_dir"] else None
            )
        else:
            rl_runs_started = state["rl_runs_started"]
            ppo_run_dir = state["run_dir"]

        ppo_report = TesterAgent().test(state["task_brief"], state["code"], ppo_config, ppo_run_dir)
        if ppo_report is not None:
            state["recorder"].write(format_ppo_live_log(ppo_report), agent="ppo", kind="output")

        test_report = _make_test_report(state["code_review"])
        record = IterationRecord(
            iteration=state["iteration"],
            plan=state["plan"],
            code=state["code"],
            code_review=state["code_review"],
            test_report=test_report,
            ppo_report=ppo_report,
            critique=None,
        )
        update = {
            "ppo_report": ppo_report,
            "status": RunStatus.PPO_TESTED.value if ppo_config.run_rl_training else RunStatus.TESTED.value,
            "rl_runs_started": rl_runs_started,
            "iterations": state["iterations"] + [record],
        }
        if ppo_report.status == "skipped":
            update.update({
                "status": RunStatus.ACCEPTED_STATIC.value,
                "final_code": state["code"],
                "final_report": test_report,
                "final_ppo_report": ppo_report,
            })
            return Command(update=update, goto=END)
        return Command(update=update, goto="critic")

    def critic_node(state: AgentState) -> Command:
        ppo_report = state["ppo_report"]
        if ppo_report is None or ppo_report.status == "skipped":
            critique = CritiqueArtifact(
                feedback="",
                should_retry=False,
                metadata={"agent": "critic", "source": "ppo_not_run", "skipped": True},
            )
        else:
            critique = agents.critic(
                client,
                state["plan"],
                state["code"],
                ppo_report,
                enable_thinking=state["enable_thinking"].get("critic", False),
                logger=state["recorder"],
            )
        last = state["iterations"][-1]
        update = {
            "critique": critique,
            "iterations": state["iterations"][:-1] + [dc_replace(last, critique=critique)],
        }
        if not critique.should_retry:
            update.update({
                "critic_feedback": None,
                "status": RunStatus.ACCEPTED_RL.value,
                "final_code": state["code"],
                "final_report": _make_test_report(state["code_review"]),
                "final_ppo_report": ppo_report,
            })
            return Command(update=update, goto=END)
        update.update({
            "critic_feedback": critique.feedback,
            "reviewer_feedback": None,
            "status": RunStatus.CRITIC_FEEDBACK.value,
        })
        if _budget_exhausted(state):
            return Command(update=update, goto="finalizer")
        return Command(update=update, goto="coder")

    def finalizer_node(state: AgentState) -> dict:
        ppo_config = state["ppo_config"]
        if (
            ppo_config.run_rl_training
            and state["code_review"] is not None
            and not state["code_review"].passed
            and state["reviewer_blocked_rl_attempts"] >= state["max_iterations"]
        ):
            base_report = _make_test_report(state["code_review"])
            return {
                "status": RunStatus.FAILED_VALIDATION.value,
                "final_code": state["code"],
                "final_report": _reviewer_blocked_report(
                    base_report, state["max_iterations"], state["reviewer_blocked_rl_attempts"]
                ),
                "final_ppo_report": None,
            }
        terminal = RunStatus.FAILED_PPO if ppo_config.run_rl_training else RunStatus.FAILED_STATIC
        if state["iterations"]:
            last = state["iterations"][-1]
            return {
                "status": terminal.value,
                "final_code": last.code,
                "final_report": last.test_report,
                "final_ppo_report": last.ppo_report,
            }
        return {"status": terminal.value}

    # ── assemble graph ────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("tester", tester_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalizer", finalizer_node)
    graph.add_edge(START, "planner")
    graph.add_edge("finalizer", END)
    return graph.compile()
