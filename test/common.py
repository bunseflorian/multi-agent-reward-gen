"""Shared helpers for the agent test scripts.

Each test hits a running vLLM server directly and avoids importing the parent
``isaaclab_tasks`` package (which pulls in heavy deps).
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from qwen_agentv2.configs import config_get

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = ROOT / "pipeline" / "prompts.py"

DEFAULT_BASE_URL = config_get("llm.base_url")
DEFAULT_MODEL = config_get("llm.model")


def load_prompt(name: str) -> str:
    """Return the body of a triple-quoted constant defined in prompts.py."""

    text = PROMPTS_FILE.read_text()
    match = re.search(rf'{name} = """(.*?)"""', text, re.DOTALL)
    if match is None:
        raise KeyError(f"{name} not found in {PROMPTS_FILE}")
    return match.group(1)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.8,
    presence_penalty: float = 0.5,
    top_k: int = 20,
    enable_thinking: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send a chat-completion request to a local vLLM server and return the text."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def print_banner(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
