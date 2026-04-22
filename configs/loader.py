"""Load and resolve central filesystem paths for qwen_agentv2.

Inputs come from ``configs/paths.yaml``. Public helpers return resolved
``Path`` objects for project, IsaacLab, model, runtime, and environment paths.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "paths.yaml"


@lru_cache(maxsize=1)
def load_raw_config() -> dict[str, Any]:
    """Return the parsed YAML mapping without resolving path strings."""

    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {CONFIG_PATH}")
    return data


@lru_cache(maxsize=1)
def load_paths() -> dict[str, Any]:
    """Load the config with all string paths resolved to absolute paths."""

    data = load_raw_config()
    roots = data.get("roots", {})
    root = roots.get("isaaclab_root") if isinstance(roots, dict) else None
    if not isinstance(root, str) or not root.strip():
        raise ValueError("paths.yaml must define roots.isaaclab_root")
    return _resolve(data, Path(root).expanduser().resolve())


def get_path(*keys: str) -> Path:
    """Return a resolved ``Path`` for a nested key sequence."""

    value = _traverse(load_paths(), keys)
    if not isinstance(value, str):
        raise TypeError(f"Configured value is not a path: {'.'.join(keys)}")
    return Path(value)


def get_raw_value(*keys: str) -> Any:
    """Return the raw configured YAML value for a nested key sequence."""

    return _traverse(load_raw_config(), keys)


def project_root() -> Path:
    """Return the qwen_agentv2 project root."""

    return get_path("project", "qwen_agent_root")


def isaaclab_root() -> Path:
    """Return the IsaacLab repository root."""

    return get_path("roots", "isaaclab_root")


def _traverse(data: Any, keys: tuple[str, ...]) -> Any:
    """Walk a nested mapping by key sequence, raising ``KeyError`` on miss."""

    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _resolve(value: Any, base_dir: Path) -> Any:
    """Resolve string path values relative to *base_dir*, recursively."""

    if isinstance(value, dict):
        return {k: _resolve(v, base_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, base_dir) for v in value]
    if isinstance(value, str):
        path = Path(value)
        return str(path if path.is_absolute() else (base_dir / path).resolve())
    return value
