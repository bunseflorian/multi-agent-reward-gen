"""Load Qwen client configuration from YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .types import QwenConfig, QwenSamplingConfig


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "llm_api_config.yaml"

# Each entry maps an environment variable to a (field_name, caster) pair.
_ENV_OVERRIDES = {
    "QWEN_BASE_URL": ("base_url", str),
    "QWEN_MODEL": ("model", str),
    "QWEN_API_KEY": ("api_key", str),
    "QWEN_TIMEOUT_SECONDS": ("timeout_seconds", float),
    "QWEN_MAX_TOKENS": ("max_tokens", int),
    "QWEN_ENABLE_THINKING": ("enable_thinking", lambda value: value.lower() in {"1", "true", "yes", "on"}),
    "QWEN_DEBUG": ("debug", lambda value: value.lower() in {"1", "true", "yes", "on"}),
}


def _load_sampling_config(data: dict, key: str) -> QwenSamplingConfig:
    raw = data.get(key) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected '{key}' to be a mapping in {CONFIG_PATH}")
    return QwenSamplingConfig(**raw)


def load_qwen_config() -> QwenConfig:
    """Load ``QwenConfig`` from YAML plus ``QWEN_*`` environment overrides."""

    data: dict = {}
    if CONFIG_PATH.is_file():
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data.update(loaded)

    for env_name, (field_name, caster) in _ENV_OVERRIDES.items():
        raw_value = os.getenv(env_name)
        if raw_value is not None:
            data[field_name] = caster(raw_value)

    allowed = set(QwenConfig.__dataclass_fields__) - {"non_reasoning", "reasoning"}
    config_data = {k: v for k, v in data.items() if k in allowed}
    config_data["non_reasoning"] = _load_sampling_config(data, "non_reasoning")
    config_data["reasoning"] = _load_sampling_config(data, "reasoning")
    return QwenConfig(**config_data)
