"""Build the ``isaaclab.sh`` RL-Games training command."""

from __future__ import annotations

from pathlib import Path

from ..configs import get_path, get_raw_value, isaaclab_root as configured_isaaclab_root
from ..core import PPOConfig


QWEN_AGENT_EXPERIMENT_NAME = "qwen_agent_task"


def build_ppo_command(
    config: PPOConfig,
    isaaclab_root: Path | None = None,
    rl_games_root: str | Path | None = None,
    experiment_name: str | None = None,
) -> list[str]:
    """Return the RL-Games training command list suitable for ``subprocess``."""

    root = Path(isaaclab_root) if isaaclab_root is not None else configured_isaaclab_root()
    launcher = root / get_path("isaaclab", "launcher").relative_to(configured_isaaclab_root())
    train_script = get_path("isaaclab", "rl_games_train_script").relative_to(configured_isaaclab_root())

    command = [
        str(launcher),
        "-p",
        str(train_script),
        "--task",
        config.task_id,
    ]
    conda_env = configured_conda_env()
    if conda_env:
        command = ["conda", "run", "-n", conda_env, *command]
    if config.num_envs is not None:
        command.extend(["--num_envs", str(config.num_envs)])
    if config.max_iterations is not None:
        command.extend(["--max_iterations", str(config.max_iterations)])
    if config.headless:
        command.append("--headless")
    if rl_games_root is not None:
        command.append(f"++params.config.train_dir={Path(rl_games_root).resolve()}")
    if experiment_name is not None:
        command.append(f"++params.config.full_experiment_name={experiment_name}")
    return command


def configured_conda_env() -> str | None:
    """Return the IsaacLab conda env from ``paths.yaml``, or ``None``."""

    try:
        conda_env = get_raw_value("isaaclab", "conda_env")
    except KeyError:
        return None
    if not isinstance(conda_env, str):
        return None
    return conda_env or None
