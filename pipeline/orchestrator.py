"""Coordinate Planner → Coder → Reviewer → Tester → Critic iterations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..core import (
    CodeArtifact,
    CodeReviewArtifact,
    CritiqueArtifact,
    IterationRecord,
    PlanArtifact,
    PPOConfig,
    PPOReport,
    RunState,
    RunStatus,
    TaskBrief,
    TestReport,
)
from ..llm import QwenApiError, QwenClient
from .coder import CoderAgent
from .critic import CriticAgent
from .formatting import format_ppo_live_log
from .logger import LiveLogger, null_logger
from .planner import PlannerAgent
from .reviewer import ReviewerAgent
from .tester import TesterAgent


_MAX_FORMAT_RETRIES = 3


class MultiAgentRewardSystem:
    """Coordinates all agent roles for iterative reward-code generation."""

    def __init__(
        self,
        client: QwenClient,
        *,
        logger: LiveLogger | None = None,
        planner: PlannerAgent | None = None,
        coder: CoderAgent | None = None,
        reviewer: ReviewerAgent | None = None,
        tester: TesterAgent | None = None,
        critic: CriticAgent | None = None,
        agent_reasoning: dict[str, bool] | None = None,
    ):
        self.client = client
        self.logger = logger or null_logger()
        reasoning = agent_reasoning or {}
        self.planner = planner or PlannerAgent(client, enable_thinking=reasoning.get("planner", False), logger=self.logger)
        self.coder = coder or CoderAgent(client, enable_thinking=reasoning.get("coder", False), logger=self.logger)
        self.reviewer = reviewer or ReviewerAgent(client, enable_thinking=reasoning.get("reviewer", False), logger=self.logger)
        self.tester = tester or TesterAgent()
        self.critic = critic or CriticAgent(client, enable_thinking=reasoning.get("critic", False), logger=self.logger)

    def run(
        self,
        task_brief: TaskBrief,
        *,
        max_iterations: int = 3,
        max_rl_iterations: int | None = None,
        ppo_config: PPOConfig | None = None,
        run_dir: str | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> RunState:
        """Run the full Planner → Coder → Reviewer → PPO → Critic loop."""

        notify = status_callback or (lambda _msg: None)
        ppo_config = ppo_config or PPOConfig()
        state = RunState(task_brief=task_brief, rl_training_skipped=not ppo_config.run_rl_training)

        notify("Planner")
        plan = self._plan_or_fail(task_brief, state)
        if plan is None:
            return state
        state.status = RunStatus.PLANNED

        loop = _IterationLoop(
            system=self,
            state=state,
            task_brief=task_brief,
            plan=plan,
            ppo_config=ppo_config,
            run_dir=run_dir,
            max_iterations=max_iterations,
            target_rl_runs=max_rl_iterations if max_rl_iterations is not None else max_iterations,
            notify=notify,
        )
        return loop.execute()

    def _plan_or_fail(self, task_brief: TaskBrief, state: RunState) -> PlanArtifact | None:
        try:
            return self.planner.plan(task_brief)
        except QwenApiError:
            raise
        except Exception as exc:
            state.status = RunStatus.FAILED_PLANNING
            state.final_report = TestReport(
                passed=False,
                errors=[f"Planner failed: {type(exc).__name__}: {exc}"],
                metadata={"agent": "planner", "stage": "planning"},
            )
            return None


class _IterationLoop:
    """Stateful helper that drives the iterative agent loop."""

    def __init__(
        self,
        *,
        system: MultiAgentRewardSystem,
        state: RunState,
        task_brief: TaskBrief,
        plan: PlanArtifact,
        ppo_config: PPOConfig,
        run_dir: str | None,
        max_iterations: int,
        target_rl_runs: int,
        notify: Callable[[str], None],
    ):
        self.system = system
        self.state = state
        self.task_brief = task_brief
        self.plan = plan
        self.ppo_config = ppo_config
        self.run_dir = run_dir
        self.max_iterations = max_iterations
        self.target_rl_runs = target_rl_runs
        self.notify = notify

        self.reviewer_feedback: str | None = None
        self.critic_feedback: str | None = None
        self.previous_code: CodeArtifact | None = None
        self.rl_runs_started = 0
        self.reviewer_blocked_rl_attempts = 0
        self.format_retries = 0
        self.iteration = 0

    # ------------------------------------------------------------------ loop

    def execute(self) -> RunState:
        while self._should_continue():
            self.iteration += 1

            try:
                code, code_review, test_report = self._generate_and_review()
                ppo_report, critique = self._test_and_critique(code, test_report)
                self.format_retries = 0
            except ValueError as exc:
                if self._handle_format_error(exc):
                    return self.state
                continue

            self.state.iterations.append(
                IterationRecord(
                    iteration=self.iteration,
                    plan=self.plan,
                    code=code,
                    code_review=code_review,
                    test_report=test_report,
                    ppo_report=ppo_report,
                    critique=critique,
                )
            )
            self.previous_code = code

            finished = self._evaluate_iteration(code, code_review, test_report, ppo_report, critique)
            if finished:
                return self.state

        return self._finalize_after_loop()

    # ---------------------------------------------------------------- helpers

    def _should_continue(self) -> bool:
        if self.ppo_config.run_rl_training:
            return self.rl_runs_started < self.target_rl_runs
        return self.iteration < self.max_iterations

    def _generate_and_review(self) -> tuple[CodeArtifact, CodeReviewArtifact, TestReport]:
        self.notify(f"Coder (iteration {self.iteration})")
        code = self.system.coder.write_code(
            self.task_brief,
            self.plan,
            previous_code=self.previous_code,
            reviewer_feedback=self.reviewer_feedback,
            critic_feedback=self.critic_feedback,
        )
        self.state.status = RunStatus.CODE_GENERATED

        self.notify(f"Reviewer (iteration {self.iteration})")
        code_review = self.system.reviewer.review(self.task_brief, self.plan, code)
        test_report = TestReport(
            passed=code_review.passed,
            errors=code_review.errors,
            warnings=code_review.warnings,
            metadata={"agent": "reviewer", "stage": "static_llm_validation"},
        )
        self.state.status = (
            RunStatus.VALIDATED_STATIC if code_review.passed else RunStatus.FAILED_VALIDATION
        )
        return code, code_review, test_report

    def _test_and_critique(
        self,
        code: CodeArtifact,
        test_report: TestReport,
    ) -> tuple[PPOReport | None, CritiqueArtifact | None]:
        if not test_report.passed:
            return None, None

        if self.ppo_config.run_rl_training:
            self.rl_runs_started += 1
            self.notify(f"PPO Training (iteration {self.iteration})")
            self.state.status = RunStatus.PPO_RUNNING
            ppo_run_dir = _ppo_run_dir(self.run_dir, self.rl_runs_started)
        else:
            self.state.status = RunStatus.TESTED
            ppo_run_dir = self.run_dir

        ppo_report = self.system.tester.test(self.task_brief, code, self.ppo_config, ppo_run_dir)
        if ppo_report is not None:
            self.system.logger.log_output(format_ppo_live_log(ppo_report), agent="ppo", extract=False)
        self.state.status = RunStatus.PPO_TESTED if self.ppo_config.run_rl_training else RunStatus.TESTED

        if ppo_report.status == "skipped":
            return ppo_report, None

        self.notify(f"Critic (iteration {self.iteration})")
        critique = self.system.critic.critique(self.task_brief, self.plan, code, ppo_report)
        return ppo_report, critique

    def _handle_format_error(self, exc: ValueError) -> bool:
        self.format_retries += 1
        if self.format_retries >= _MAX_FORMAT_RETRIES:
            self.state.status = RunStatus.FAILED_FORMAT
            self.state.final_report = TestReport(
                passed=False,
                errors=[f"Output format error: {exc}"],
                metadata={"stage": "format_check", "format_retries": self.format_retries},
            )
            return True
        self.notify(f"Format error (retry {self.format_retries}/{_MAX_FORMAT_RETRIES}): {exc}")
        return False

    def _evaluate_iteration(
        self,
        code: CodeArtifact,
        code_review: CodeReviewArtifact,
        test_report: TestReport,
        ppo_report: PPOReport | None,
        critique: CritiqueArtifact | None,
    ) -> bool:
        if not test_report.passed:
            self.reviewer_feedback = code_review.feedback
            self.critic_feedback = None
            if self.ppo_config.run_rl_training:
                self.reviewer_blocked_rl_attempts += 1
                if self.reviewer_blocked_rl_attempts >= self.max_iterations:
                    self.state.status = RunStatus.FAILED_VALIDATION
                    _finalize_state(
                        self.state,
                        code,
                        _reviewer_blocked_report(
                            test_report,
                            self.max_iterations,
                            self.reviewer_blocked_rl_attempts,
                        ),
                        None,
                    )
                    return True
            return False
        self.reviewer_blocked_rl_attempts = 0

        if ppo_report is not None and ppo_report.status == "skipped":
            self.state.status = RunStatus.ACCEPTED_STATIC
            _finalize_state(self.state, code, test_report, ppo_report)
            return True

        if ppo_report is not None and critique is not None and not critique.should_retry:
            self.state.status = RunStatus.ACCEPTED_RL
            _finalize_state(self.state, code, test_report, ppo_report)
            return True

        if self.ppo_config.run_rl_training and self.rl_runs_started >= self.target_rl_runs:
            return False

        self.state.status = RunStatus.CRITIC_FEEDBACK
        self.reviewer_feedback = None
        self.critic_feedback = critique.feedback if critique is not None else None
        return False

    def _finalize_after_loop(self) -> RunState:
        self.state.status = (
            RunStatus.FAILED_PPO if self.ppo_config.run_rl_training else RunStatus.FAILED_STATIC
        )
        if self.state.iterations:
            last = self.state.iterations[-1]
            _finalize_state(self.state, last.code, last.test_report, last.ppo_report)
        return self.state


def _finalize_state(
    state: RunState,
    code: CodeArtifact,
    report: TestReport,
    ppo_report: PPOReport | None,
) -> None:
    state.final_code = code
    state.final_report = report
    state.final_ppo_report = ppo_report


def _reviewer_blocked_report(
    test_report: TestReport,
    max_iterations: int,
    reviewer_blocked_rl_attempts: int,
) -> TestReport:
    return TestReport(
        passed=False,
        errors=[
            f"RL training could not start because the reviewer rejected {max_iterations} generated candidates.",
            *test_report.errors,
        ],
        warnings=test_report.warnings,
        metadata={**test_report.metadata, "reviewer_blocked_rl_attempts": reviewer_blocked_rl_attempts},
    )


def _ppo_run_dir(run_dir: str | None, rl_run_index: int) -> str | None:
    if run_dir is None:
        return None
    return str(Path(run_dir) / f"rl_run_{rl_run_index:02d}")
