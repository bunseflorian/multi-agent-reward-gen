"""Server settings dataclass and environment-variable loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..configs import config_get, config_path


@dataclass(frozen=True)
class QwenServerSettings:
    """Settings needed to launch the local Qwen server."""

    project_root: Path
    model_path: Path
    served_model_name: str
    reasoning_parser: str
    reasoning_config: str
    host: str
    port: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    startup_timeout_seconds: int
    pid_file: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def load_qwen_server_settings(project_root: Path | None = None) -> QwenServerSettings:
    """Load server settings from central app config and ``QWEN_*`` env overrides."""

    root = project_root or config_path("paths.qwen_agent_root")
    server = config_get("server", {})
    reasoning_config = server["reasoning_config"]
    json.loads(reasoning_config)
    return QwenServerSettings(
        project_root=root,
        model_path=config_path("paths.qwen_model"),
        served_model_name=server["served_model_name"],
        reasoning_parser=server["reasoning_parser"],
        reasoning_config=reasoning_config,
        host=server["host"],
        port=int(server["port"]),
        max_model_len=int(server["max_model_len"]),
        gpu_memory_utilization=float(server["gpu_memory_utilization"]),
        max_num_seqs=int(server["max_num_seqs"]),
        startup_timeout_seconds=int(server["startup_timeout_seconds"]),
        pid_file=config_path("paths.qwen_server_pid_file"),
    )
