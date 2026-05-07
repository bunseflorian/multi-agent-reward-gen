"""Build, start, stop, and health-check the Qwen server from the UI."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from ..configs import config_get, config_path
from ..server import QwenServerSettings, qwen_server_is_ready


PROJECT_ROOT = config_path("paths.qwen_agent_root")
PACKAGE_IMPORT_ROOT = PROJECT_ROOT.parent
SERVER_PID_FILE = config_path("paths.qwen_server_pid_file")
SERVER_LOG_FILE = config_path("paths.qwen_server_log_file")


def build_server_command(
    settings: QwenServerSettings,
    pid_file: Path = SERVER_PID_FILE,
) -> list[str]:
    """Return the command used to launch the Qwen server wrapper."""

    return [
        *_server_python_command(),
        "-m", "qwen_agentv2.server.launcher",
        "--model-path", str(settings.model_path),
        "--served-model-name", settings.served_model_name,
        "--reasoning-parser", settings.reasoning_parser,
        "--reasoning-config", settings.reasoning_config,
        "--host", settings.host,
        "--port", str(settings.port),
        "--max-model-len", str(settings.max_model_len),
        "--gpu-memory-utilization", str(settings.gpu_memory_utilization),
        "--max-num-seqs", str(settings.max_num_seqs),
        "--pid-file", str(pid_file),
    ]


def launch_in_background(
    command: list[str],
    cwd: Path = PACKAGE_IMPORT_ROOT,
    log_file: Path = SERVER_LOG_FILE,
) -> subprocess.Popen[bytes]:
    """Start a background process with output appended to *log_file*."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("ab")
    try:
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=_server_env(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def read_server_pid(pid_file: Path = SERVER_PID_FILE) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def stop_server(pid_file: Path = SERVER_PID_FILE) -> int | None:
    pid = read_server_pid(pid_file)
    if pid is None:
        return None
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)
    return pid


def wait_for_server(base_url: str, timeout_seconds: int, *, ready: bool) -> bool:
    """Poll until the server matches the desired *ready* state, or timeout."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if qwen_server_is_ready(base_url, timeout=2.0 if ready else 1.0) == ready:
            return True
        time.sleep(1 if not ready else 2)
    return False


def read_log_tail(log_file: Path = SERVER_LOG_FILE, max_chars: int = 6000) -> str:
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return text[-max_chars:]


def _server_python_command() -> list[str]:
    """Resolve the Python interpreter used to launch the server."""

    explicit = os.environ.get("QWEN_SERVER_PYTHON")
    if explicit:
        return [explicit]
    conda_env = config_get("paths.qwen_server_conda_env")
    if not conda_env:
        raise ValueError("Missing config value: paths.qwen_server_conda_env")
    return ["conda", "run", "-n", conda_env, "python"]


def _server_env() -> dict[str, str]:
    """Ensure the qwen_agentv2 package is importable for child processes."""

    env = os.environ.copy()
    import_root = str(PACKAGE_IMPORT_ROOT)
    current = env.get("PYTHONPATH")
    if not current:
        env["PYTHONPATH"] = import_root
    else:
        paths = current.split(os.pathsep)
        if import_root not in paths:
            env["PYTHONPATH"] = os.pathsep.join([import_root, current])
    return env
