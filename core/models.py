"""Dataclasses and enums shared by the multi-agent reward system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RunStatus(str, Enum):
    """Lifecycle states for a multi-agent reward-generation run."""

    CREATED = "created"
    PLANNED = "planned"
    CODE_GENERATED = "code_generated"
    VALIDATED_STATIC = "validated_static"
    PPO_RUNNING = "ppo_running"
    PPO_TESTED = "ppo_tested"
    TESTED = "tested"
    CRITIC_FEEDBACK = "critic_feedback"
    ACCEPTED_STATIC = "accepted_static"
    ACCEPTED_RL = "accepted_rl"
    FAILED_VALIDATION = "failed_validation"
    FAILED_PPO = "failed_ppo"
    FAILED_STATIC = "failed_static"
    FAILED_PLANNING = "failed_planning"
    FAILED_FORMAT = "failed_format"


@dataclass(frozen=True)
class TaskBrief:
    """Task and pipeline context passed to all agents."""

    task_id: str
    objective: str | None
    env_source_path: str
    env_code_context: str


@dataclass(frozen=True)
class PlanArtifact:
    """Planner output and extracted implementation requirements."""

    text: str
    requirements: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeArtifact:
    """Parsed coder output and model response metadata."""

    code: str
    raw_response: str
    thinking: str | None = None
    parse_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeReviewArtifact:
    """LLM review result for a generated reward-code candidate."""

    passed: bool
    feedback: str
    raw_response: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestReport:
    """Static validation report for generated reward code."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Keeps pytest from collecting the dataclass as a test container.
TestReport.__test__ = False


@dataclass(frozen=True)
class PPOConfig:
    """Runtime settings for optional PPO validation."""

    run_rl_training: bool = False
    num_envs: int | None = None
    max_iterations: int | None = None
    timeout_seconds: int | None = None
    headless: bool = True
    task_id: str = "qwen_agent_task"
    backend: str = "rl_games"
    progress_callback: Callable[[dict[str, Any]], None] | None = field(default=None, compare=False, repr=False)
    pre_training_callback: Callable[[], None] | None = field(default=None, compare=False, repr=False)
    post_training_callback: Callable[[], None] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class PPOReport:
    """Structured result of optional PPO validation."""

    status: str
    passed: bool
    command: list[str] = field(default_factory=list)
    log_path: str | None = None
    summary_path: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CritiqueArtifact:
    """Critic feedback and retry decision."""

    feedback: str
    should_retry: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IterationRecord:
    """One planner/coder/validator/tester/critic iteration."""

    iteration: int
    plan: PlanArtifact
    code: CodeArtifact
    code_review: CodeReviewArtifact | None
    test_report: TestReport
    ppo_report: PPOReport | None = None
    critique: CritiqueArtifact | None = None


@dataclass
class RunState:
    """Aggregate state for one multi-agent run."""

    task_brief: TaskBrief
    status: RunStatus = RunStatus.CREATED
    iterations: list[IterationRecord] = field(default_factory=list)
    final_code: CodeArtifact | None = None
    final_report: TestReport | None = None
    final_ppo_report: PPOReport | None = None
    rl_training_skipped: bool = True

    @property
    def accepted(self) -> bool:
        """Return whether the final run status represents acceptance."""

        return self.status in {RunStatus.ACCEPTED_STATIC, RunStatus.ACCEPTED_RL}
