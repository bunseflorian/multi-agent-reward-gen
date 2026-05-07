"""Compatibility loader for Qwen client configuration."""

from __future__ import annotations

from ..configs import config_get
from .types import QwenConfig, QwenSamplingConfig


def load_qwen_config() -> QwenConfig:
    """Load ``QwenConfig`` from YAML plus ``QWEN_*`` environment overrides."""

    llm = config_get("llm", {})
    return QwenConfig(
        base_url=llm["base_url"],
        model=llm["model"],
        api_key=llm["api_key"],
        timeout_seconds=float(llm["timeout_seconds"]),
        max_tokens=int(llm["max_tokens"]),
        enable_thinking=bool(llm["enable_thinking"]),
        non_reasoning=QwenSamplingConfig(**llm["non_reasoning"]),
        reasoning=QwenSamplingConfig(**llm["reasoning"]),
        debug=bool(llm["debug"]),
    )
