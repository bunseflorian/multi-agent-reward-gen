"""Build, start, stop, and health-check the Qwen server from the UI."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import yaml

from ..configs import get_path, get_raw_value, project_root
from ..server import QwenServerSettings, qwen_server_is_ready


PROJECT_ROOT = project_root()
PACKAGE_IMPORT_ROOT = PROJECT_ROOT.parent
SERVER_PID_FILE = get_path("runtime", "qwen_server_pid_file")
SERVER_LOG_FILE = get_path("runtime", "qwen_server_log_file")
SERVER_SETTINGS_FILE = PROJECT_ROOT / ".qwen_server_ui_settings.yaml"


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


def load_persisted_server_settings(defaults: QwenServerSettings) -> QwenServerSettings:
    """Return persisted UI settings if present, otherwise *defaults*."""

    try:
        loaded = yaml.safe_load(SERVER_SETTINGS_FILE.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return defaults

    if not isinstance(loaded, dict):
        return defaults

    def _value(name: str, fallback):
        return loaded.get(name, fallback)

    model_path = Path(str(_value("model_path", defaults.model_path))).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    return QwenServerSettings(
        project_root=defaults.project_root,
        model_path=model_path,
        served_model_name=str(_value("served_model_name", defaults.served_model_name)),
        host=str(_value("host", defaults.host)),
        port=int(_value("port", defaults.port)),
        max_model_len=int(_value("max_model_len", defaults.max_model_len)),
        gpu_memory_utilization=float(_value("gpu_memory_utilization", defaults.gpu_memory_utilization)),
        max_num_seqs=int(_value("max_num_seqs", defaults.max_num_seqs)),
        startup_timeout_seconds=defaults.startup_timeout_seconds,
        pid_file=defaults.pid_file,
    )


def persist_server_settings(settings: QwenServerSettings) -> None:
    """Persist the UI-selected server settings for the next UI session."""

    payload = {
        "model_path": str(settings.model_path),
        "served_model_name": settings.served_model_name,
        "host": settings.host,
        "port": settings.port,
        "max_model_len": settings.max_model_len,
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "max_num_seqs": settings.max_num_seqs,
    }
    SERVER_SETTINGS_FILE.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def reset_persisted_server_settings() -> None:
    """Remove persisted UI server settings so defaults are used again."""

    SERVER_SETTINGS_FILE.unlink(missing_ok=True)


def _server_python_command() -> list[str]:
    """Resolve the Python interpreter used to launch the server."""

    explicit = os.environ.get("QWEN_SERVER_PYTHON")
    if explicit:
        return [explicit]
    conda_env = os.environ.get("QWEN_SERVER_CONDA_ENV")
    if not conda_env:
        try:
            conda_env = str(get_raw_value("runtime", "qwen_server_conda_env"))
        except KeyError:
            conda_env = "qwen-vllm-server"
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
