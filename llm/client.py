"""OpenAI-compatible HTTP client for a local Qwen chat-completions server."""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from .config import load_qwen_config
from .types import QwenApiError, QwenConfig, QwenResponseParseError, QwenSamplingConfig, QwenTimeoutError

logger = logging.getLogger(__name__)

Message = dict[str, str]
Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class QwenClient:
    """Client for local OpenAI-compatible Qwen chat-completions servers."""

    def __init__(self, config: QwenConfig | None = None, transport: Transport | None = None):
        self.config = config or load_qwen_config()
        self._transport = transport or _urllib_transport

    def generate(
        self,
        messages: Sequence[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Send chat messages and return the assistant content string."""

        payload = self._build_payload(messages, temperature, top_p, max_tokens, enable_thinking)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        logger.info(
            "Calling Qwen chat completions model=%s base_url=%s max_tokens=%s temperature=%s",
            self.config.model,
            self.config.base_url,
            payload["max_tokens"],
            payload["temperature"],
        )
        if self.config.debug:
            logger.debug("Qwen request payload: %s", payload)

        try:
            response = self._transport(url, headers, payload, self.config.timeout_seconds)
        except TimeoutError as exc:
            raise QwenTimeoutError(f"Qwen request timed out after {self.config.timeout_seconds}s") from exc
        except QwenApiError:
            raise
        except Exception as exc:
            raise QwenApiError(f"Qwen request failed: {exc}") from exc

        if self.config.debug:
            logger.debug("Qwen response payload: %s", response)
        return _extract_chat_content(response)

    def generate_text(self, system_prompt: str, user_prompt: str, **overrides: Any) -> str:
        """Generate text from a single system + user prompt pair."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(
            messages,
            temperature=overrides.get("temperature"),
            top_p=overrides.get("top_p"),
            max_tokens=overrides.get("max_tokens"),
            enable_thinking=overrides.get("enable_thinking"),
        )

    def _build_payload(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        enable_thinking: bool | None,
    ) -> dict[str, Any]:
        cfg = self.config
        thinking_enabled = cfg.enable_thinking if enable_thinking is None else enable_thinking
        sampling = self._sampling_config(thinking_enabled)
        payload = {
            "model": cfg.model,
            "messages": list(messages),
            "temperature": sampling.temperature if temperature is None else temperature,
            "top_p": sampling.top_p if top_p is None else top_p,
            "top_k": sampling.top_k,
            "presence_penalty": sampling.presence_penalty,
            "max_tokens": sampling.max_tokens if max_tokens is None else max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": thinking_enabled,
            },
        }
        if thinking_enabled and sampling.thinking_budget is not None:
            payload["thinking_token_budget"] = sampling.thinking_budget
        return payload

    def _sampling_config(self, enable_thinking: bool) -> QwenSamplingConfig:
        return self.config.reasoning if enable_thinking else self.config.non_reasoning


def _extract_chat_content(response: dict[str, Any]) -> str:
    """Extract assistant content from a chat-completions response."""

    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QwenResponseParseError("Malformed chat-completions response from Qwen server") from exc
    if not isinstance(content, str):
        raise QwenResponseParseError("Qwen response content is not a string")
    if choice.get("finish_reason") == "length":
        logger.warning("Qwen response was truncated because max_tokens was reached.")
    return content


def _urllib_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one JSON HTTP request through urllib."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except socket.timeout as exc:
        raise TimeoutError from exc
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise QwenApiError(f"Qwen server returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            raise TimeoutError from exc
        raise QwenApiError(f"Could not reach Qwen server: {exc.reason}") from exc

    try:
        loaded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise QwenResponseParseError("Qwen server returned invalid JSON") from exc
    if not isinstance(loaded, dict):
        raise QwenResponseParseError("Qwen server returned a non-object JSON response")
    return loaded
