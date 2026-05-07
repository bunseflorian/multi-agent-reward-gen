"""Build the ``isaaclab.sh`` RL-Games training command."""

from __future__ import annotations

from pathlib import Path

from ..configs import config_get, config_path
from ..core import PPOConfig


QWEN_AGENT_EXPERIMENT_NAME = "qwen_agent_task"


def build_ppo_command(
    config: PPOConfig,
    isaaclab_root: Path | None = None,
    rl_games_root: str | Path | None = None,
    experiment_name: str | None = None,
) -> list[str]:
    """Return the RL-Games training command list suitable for ``subprocess``."""

    isaaclab_config_root = config_path("paths.isaaclab_root")
    root = Path(isaaclab_root) if isaaclab_root is not None else isaaclab_config_root
    launcher = root / config_path("paths.isaaclab_launcher").relative_to(isaaclab_config_root)
    train_script = config_path("paths.rl_games_train_script").relative_to(isaaclab_config_root)

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
    """Return the IsaacLab conda env from central app config, or ``None``."""

    try:
        return config_get("paths.isaaclab_conda_env")
    except ValueError:
        return None
