"""Single high-level ``run_task`` entry used by the UI."""

from __future__ import annotations

from pathlib import Path

from ..configs import build_ppo_config, config_get, config_path
from ..core import PlanArtifact, PPOConfig, RunState, RunStatus, TaskBrief
from ..llm import QwenClient, load_qwen_config
from .env_context import load_env_context
from .logger import RunRecorder
from .orchestrator import build_graph


def build_initial_state(
    *,
    task_brief: TaskBrief,
    ppo_config: PPOConfig,
    max_iterations: int,
    max_rl_iterations: int | None,
    run_dir: str | None,
    agent_reasoning: dict[str, bool] | None,
    initial_plan: PlanArtifact | None = None,
    recorder: RunRecorder | None = None,
    stop_after_plan: bool = False,
) -> dict:
    """Return the AgentState dict ready to pass to graph.stream() or graph.invoke()."""
    max_format_retries = int(config_get("pipeline.max_format_retries"))
    reasoning = agent_reasoning or {}

    if initial_plan is not None:
        initial_status = RunStatus.PLANNED.value
        effective_stop_after_plan = False
        initial_plan_state = initial_plan
    else:
        initial_status = RunStatus.CREATED.value
        effective_stop_after_plan = stop_after_plan
        initial_plan_state = None

    return {
        "task_brief": task_brief,
        "ppo_config": ppo_config,
        "max_iterations": max_iterations,
        "max_rl_iterations": max_rl_iterations,
        "max_format_retries": max_format_retries,
        "stop_after_plan": effective_stop_after_plan,
        "run_dir": run_dir,
        "agent_reasoning": reasoning,
        "enable_thinking": reasoning,
        "plan": initial_plan_state,
        "code": None,
        "code_review": None,
        "ppo_report": None,
        "critique": None,
        "reviewer_feedback": None,
        "critic_feedback": None,
        "iteration": 0,
        "rl_runs_started": 0,
        "reviewer_blocked_rl_attempts": 0,
        "format_retries": 0,
        "status": initial_status,
        "final_code": None,
        "final_report": None,
        "final_ppo_report": None,
        "iterations": [],
        "recorder": recorder,
    }


def run_task(
    *,
    max_iterations: int | None = None,
    max_rl_iterations: int | None = None,
    client: QwenClient | None = None,
    env_path: str | Path | None = None,
    objective: str | None = None,
    ppo_config: PPOConfig | None = None,
    run_dir: str | Path | None = None,
    agent_reasoning: dict[str, bool] | None = None,
    stop_after_plan: bool = False,
    initial_plan: PlanArtifact | None = None,
    randomize_env: bool = False,
) -> RunState:
    """Run one configured reward-generation task end-to-end."""

    resolved_env = env_path or config_path("paths.default_env")

    if randomize_env and run_dir:
        from .env_randomizer import create_candidate_env
        effective_env_path = create_candidate_env(Path(resolved_env), Path(run_dir))
    else:
        effective_env_path = resolved_env

    task_brief = _task_from_env_path(effective_env_path, config_path("paths.qwen_agent_root"), objective=objective)

    recorder = RunRecorder(Path(run_dir)) if run_dir else RunRecorder(None)
    ppo_config = ppo_config or build_ppo_config()
    qwen_client = client or QwenClient(load_qwen_config())

    max_iters = max_iterations if max_iterations is not None else int(config_get("pipeline.max_iterations"))
    _raw_rl = max_rl_iterations if max_rl_iterations is not None else config_get("pipeline.max_rl_iterations")
    max_rl_iters = int(_raw_rl) if _raw_rl is not None else max_iters

    initial_state = build_initial_state(
        task_brief=task_brief,
        ppo_config=ppo_config,
        max_iterations=max_iters,
        max_rl_iterations=max_rl_iters,
        run_dir=str(run_dir) if run_dir else None,
        agent_reasoning=agent_reasoning,
        initial_plan=initial_plan,
        recorder=recorder,
        stop_after_plan=stop_after_plan,
    )

    graph = build_graph(qwen_client, recorder=recorder)
    final_state = graph.invoke(initial_state)

    return _state_to_run_state(final_state, task_brief)


def _state_to_run_state(state: dict, task_brief: TaskBrief) -> RunState:
    run_state = RunState(
        task_brief=task_brief,
        status=RunStatus(state["status"]),
        iterations=state.get("iterations", []),
        plan=state.get("plan"),
        final_code=state.get("final_code"),
        final_report=state.get("final_report"),
        final_ppo_report=state.get("final_ppo_report"),
        rl_training_skipped=not state["ppo_config"].run_rl_training,
    )
    run_state.recorder = state.get("recorder")
    return run_state


def _task_from_env_path(
    env_path: str | Path,
    root: Path | None = None,
    *,
    objective: str | None = None,
) -> TaskBrief:
    env_source_path, env_code_context, (_env_class, _cfg_class) = load_env_context(env_path, root)
    return TaskBrief(
        task_id=Path(env_path).stem,
        objective=objective,
        env_source_path=str(env_source_path),
        env_code_context=env_code_context,
    )
