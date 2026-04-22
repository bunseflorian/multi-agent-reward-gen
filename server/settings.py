"""Server settings dataclass and environment-variable loader."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..configs import get_path, project_root as configured_project_root


@dataclass(frozen=True)
class QwenServerSettings:
    """Settings needed to launch the local Qwen server."""

    project_root: Path
    model_path: Path
    served_model_name: str = "Qwen/Qwen3.5-4B"
    reasoning_parser: str = "qwen3"
    reasoning_config: str = '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
    host: str = "127.0.0.1"
    port: int = 8000
    max_model_len: int = 12288
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 1
    startup_timeout_seconds: int = 300
    pid_file: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def load_qwen_server_settings(project_root: Path | None = None) -> QwenServerSettings:
    """Load server settings from ``paths.yaml`` and ``QWEN_*`` env overrides."""

    root = project_root or configured_project_root()
    model_path = Path(os.getenv("QWEN_MODEL_PATH", str(get_path("models", "qwen")))).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path

    pid_file_env = os.getenv("QWEN_SERVER_PID_FILE")
    reasoning_config = os.getenv(
        "QWEN_REASONING_CONFIG",
        '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
    )
    json.loads(reasoning_config)
    return QwenServerSettings(
        project_root=root,
        model_path=model_path,
        served_model_name=os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-4B"),
        reasoning_parser=os.getenv("QWEN_REASONING_PARSER", "qwen3"),
        reasoning_config=reasoning_config,
        host=os.getenv("QWEN_HOST", "127.0.0.1"),
        port=int(os.getenv("QWEN_PORT", "8000")),
        max_model_len=int(os.getenv("QWEN_MAX_MODEL_LEN", "11776")),
        gpu_memory_utilization=float(os.getenv("QWEN_GPU_MEMORY_UTILIZATION", "0.85")),
        max_num_seqs=int(os.getenv("QWEN_MAX_NUM_SEQS", "1")),
        startup_timeout_seconds=int(os.getenv("QWEN_STARTUP_TIMEOUT_SECONDS", "300")),
        pid_file=Path(pid_file_env).expanduser() if pid_file_env else None,
    )
