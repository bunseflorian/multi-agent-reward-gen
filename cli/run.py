"""Single high-level ``run_task`` entry used by CLI and UI alike."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..configs import get_path, project_root
from ..core import PPOConfig, RunState, TaskBrief
from ..llm import QwenClient, load_qwen_config
from ..pipeline import LiveLogger, MultiAgentRewardSystem
from ..pipeline.env_context import load_env_context


def run_task(
    *,
    max_iterations: int = 3,
    max_rl_iterations: int | None = None,
    client: QwenClient | None = None,
    env_path: str | Path | None = None,
    objective: str | None = None,
    ppo_config: PPOConfig | None = None,
    run_dir: str | Path | None = None,
    agent_reasoning: dict[str, bool] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> RunState:
    """Run one configured reward-generation task end-to-end."""

    resolved_env = env_path or get_path("environment", "default_env")
    task_brief = _task_from_env_path(resolved_env, project_root(), objective=objective)

    logger = LiveLogger(Path(run_dir)) if run_dir else LiveLogger(None)
    system = MultiAgentRewardSystem(
        client or QwenClient(load_qwen_config()),
        logger=logger,
        agent_reasoning=agent_reasoning,
    )
    return system.run(
        task_brief,
        max_iterations=max_iterations,
        max_rl_iterations=max_rl_iterations,
        ppo_config=ppo_config,
        run_dir=str(run_dir) if run_dir else None,
        status_callback=status_callback,
    )


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
