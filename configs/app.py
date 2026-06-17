"""Small central config loader.

The app uses one YAML file plus environment overrides. The public return value
is a plain dict; callers use ``config_get("section.key")`` and
``config_path("paths.some_path")`` for the few conveniences we actually need.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
APP_CONFIG_PATH = CONFIG_DIR / "app.yaml"


def _as_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def _optional_str(value: str) -> str | None:
    return value or None


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


_ENV_OVERRIDES: dict[str, tuple[str, Callable[[str], Any]]] = {
    "QWEN_ISAACLAB_ROOT": ("paths.isaaclab_root", str),
    "QWEN_PROJECT_ROOT": ("paths.qwen_agent_root", str),
    "QWEN_DEFAULT_ENV": ("paths.default_env", str),
    "QWEN_ENV_DIR": ("paths.env_dir", str),
    "QWEN_RUNS_DIR": ("paths.runs_dir", str),
    "QWEN_PPO_CANDIDATE_DIR": ("paths.ppo_candidate_dir", str),
    "QWEN_ISAACLAB_LAUNCHER": ("paths.isaaclab_launcher", str),
    "QWEN_RL_GAMES_TRAIN_SCRIPT": ("paths.rl_games_train_script", str),
    "QWEN_ISAACLAB_CONDA_ENV": ("paths.isaaclab_conda_env", _optional_str),
    "QWEN_SERVER_LAUNCHER": ("server.launcher", _optional_str),
    "QWEN_SERVER_PID_FILE": ("server.pid_file", _optional_str),
    "QWEN_SERVER_STARTUP_TIMEOUT": ("server.startup_timeout_seconds", int),
    "QWEN_BASE_URL": ("llm.base_url", str),
    "QWEN_MODEL": ("llm.model", str),
    "QWEN_API_KEY": ("llm.api_key", str),
    "QWEN_TIMEOUT_SECONDS": ("llm.timeout_seconds", float),
    "QWEN_MAX_TOKENS": ("llm.max_tokens", int),
    "QWEN_ENABLE_THINKING": ("llm.enable_thinking", _as_bool),
    "QWEN_DEBUG": ("llm.debug", _as_bool),
    "QWEN_MAX_ITERATIONS": ("pipeline.max_iterations", int),
    "QWEN_MAX_RL_ITERATIONS": ("pipeline.max_rl_iterations", _optional_int),
    "QWEN_RUN_RL_TRAINING": ("ppo.run_rl_training", _as_bool),
    "QWEN_PPO_NUM_ENVS": ("ppo.num_envs", _optional_int),
    "QWEN_PPO_MAX_ITERATIONS": ("ppo.max_iterations", _optional_int),
    "QWEN_PPO_TIMEOUT_SECONDS": ("ppo.timeout_seconds", _optional_int),
    "QWEN_PPO_HEADLESS": ("ppo.headless", _as_bool),
    "QWEN_PPO_TASK_ID": ("ppo.task_id", str),
    "QWEN_PPO_BACKEND": ("ppo.backend", str),
    "QWEN_UI_OBJECTIVE": ("ui.objective", str),
    "QWEN_UI_MAX_STATIC_ITERATIONS": ("ui.max_static_iterations", int),
    "QWEN_UI_MAX_RL_REVIEWER_RETRIES": ("ui.max_rl_reviewer_retries", int),
    "QWEN_UI_MAX_RL_ITERATIONS": ("ui.max_rl_iterations", int),
}


@lru_cache(maxsize=1)
def load_app_config() -> dict[str, Any]:
    """Return app config as a plain dict with environment overrides applied."""

    config = _read_yaml_mapping(APP_CONFIG_PATH)
    for env_name, (key_path, caster) in _ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value is not None:
            _set_nested(config, key_path, caster(value))
    _resolve_paths(config)
    return config


def clear_app_config_cache() -> None:
    load_app_config.cache_clear()


def config_get(key_path: str, default: Any = None) -> Any:
    """Read a dotted config key from ``load_app_config()``."""

    return _get_nested(load_app_config(), key_path, default)


def config_path(key_path: str) -> Path:
    """Read a dotted config key and return it as a resolved ``Path``."""

    value = config_get(key_path)
    if value is None:
        raise KeyError(key_path)
    return Path(value)


def raw_path_config() -> dict[str, Any]:
    """Return the path section in the old nested shape used by compatibility APIs."""

    paths = load_app_config()["paths"]
    return {
        "roots": {"isaaclab_root": paths["isaaclab_root"]},
        "project": {"qwen_agent_root": paths["qwen_agent_root"]},
        "environment": {"default_env": paths["default_env"], "env_dir": paths["env_dir"]},
        "runtime": {
            "runs_dir": paths["runs_dir"],
            "ppo_candidate_dir": paths["ppo_candidate_dir"],
        },
        "isaaclab": {
            "conda_env": paths.get("isaaclab_conda_env"),
            "launcher": paths["isaaclab_launcher"],
            "rl_games_train_script": paths["rl_games_train_script"],
        },
    }


def build_ppo_config(**overrides: Any):
    """Create ``PPOConfig`` from central defaults plus explicit overrides."""

    from ..core import PPOConfig

    payload = dict(config_get("ppo", {}))
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return PPOConfig(**payload)


def resolve_path_value(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}")
    return data


def _resolve_paths(config: dict[str, Any]) -> None:
    paths = config.setdefault("paths", {})
    root_raw = paths.get("isaaclab_root")
    if not root_raw:
        raise ValueError("app.yaml must define paths.isaaclab_root")
    root = Path(root_raw).expanduser().resolve()
    paths["isaaclab_root"] = str(root)
    for key, value in list(paths.items()):
        if key.endswith("_conda_env") or value is None:
            continue
        paths[key] = str(resolve_path_value(value, root))
    server = config.setdefault("server", {})
    for key in ("launcher", "pid_file"):
        value = server.get(key)
        if value is not None:
            server[key] = str(resolve_path_value(value, root))


def _get_nested(data: dict[str, Any], key_path: str, default: Any = None) -> Any:
    value: Any = data
    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _set_nested(data: dict[str, Any], key_path: str, value: Any) -> None:
    current = data
    parts = key_path.split(".")
    for key in parts[:-1]:
        current = current.setdefault(key, {})
    current[parts[-1]] = value
