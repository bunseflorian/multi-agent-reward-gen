"""Check whether the local Qwen OpenAI-compatible server is ready."""

from __future__ import annotations

import os
import urllib.request


def qwen_server_is_ready(base_url: str, *, timeout: float = 5.0) -> bool:
    """Return ``True`` when ``<base_url>/models`` responds without error."""

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": "Bearer " + os.getenv("QWEN_API_KEY", "EMPTY")},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status < 400
    except Exception:
        return False
