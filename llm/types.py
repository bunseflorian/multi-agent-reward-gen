"""Shared typed data and exceptions for the Qwen API layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenSamplingConfig:
    """Sampling settings for one generation mode."""

    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_tokens: int
    thinking_budget: int | None = None


@dataclass(frozen=True)
class QwenConfig:
    """Runtime settings for an OpenAI-compatible Qwen server.

    Values are populated by :func:`qwen_agentv2.llm.config.load_qwen_config`
    from ``configs/llm_api_config.yaml`` plus ``QWEN_*`` environment overrides.
    """

    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    max_tokens: int
    enable_thinking: bool
    non_reasoning: QwenSamplingConfig
    reasoning: QwenSamplingConfig
    debug: bool


class QwenApiError(RuntimeError):
    """Raised when the Qwen-compatible API returns an error or malformed response."""


class QwenTimeoutError(QwenApiError):
    """Raised when the Qwen-compatible API request times out."""


class QwenResponseParseError(QwenApiError):
    """Raised when a Qwen-compatible API response cannot be parsed."""
