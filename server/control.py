"""Stop and start the local vLLM server around PPO training."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from .health import qwen_server_is_ready

logger = logging.getLogger(__name__)


def stop_server(pid_file: Path, base_url: str, timeout: int = 60) -> bool:
    """Kill the server, then wait for the GPU to actually reflect it before returning."""
    pid = _read_pid(pid_file)
    if pid is None:
        logger.warning("No PID file at %s — server was not started via the configured launcher.", pid_file)
        return False

    logger.info("Killing vLLM server pid=%s", pid)
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        except OSError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break

    pid_file.unlink(missing_ok=True)
    wait_for_gpu_memory_released()
    logger.info("vLLM server killed — GPU memory released.")
    return True


def wait_for_gpu_memory_released(timeout: float = 15.0, interval: float = 0.5, threshold_mib: int = 500) -> None:
    """Poll ``nvidia-smi`` until GPU memory usage drops near idle or ``timeout`` elapses."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            used_mib = int(result.stdout.strip().splitlines()[0])
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, IndexError):
            return  # nvidia-smi unavailable or unparsable -- skip the extra wait
        if used_mib <= threshold_mib:
            return
        time.sleep(interval)
    logger.warning("GPU memory still elevated after %.0fs wait; proceeding anyway.", timeout)


def start_server(launcher: Path, pid_file: Path, base_url: str, timeout: int = 300) -> bool:
    """Launch the server in the background and wait until it responds.

    Returns True when the server is ready, False on timeout.
    """
    log_file = launcher.parent / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting vLLM server via %s", launcher)
    command = _build_launcher_command(launcher, pid_file)
    with log_file.open("wb") as lf:
        subprocess.Popen(
            command,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    if _wait_until(lambda: qwen_server_is_ready(base_url, timeout=2.0), timeout):
        logger.info("vLLM server ready at %s", base_url)
        return True

    logger.warning("Server did not become ready within %ss.", timeout)
    return False


def _build_launcher_command(launcher: Path, pid_file: Path) -> list[str]:
    try:
        from ..configs import config_get
        conda_env = config_get("server.conda_env")
    except (ValueError, KeyError):
        conda_env = None
    python_cmd = ["conda", "run", "--no-capture-output", "-n", conda_env, "python"] if conda_env else ["python"]
    return [*python_cmd, str(launcher), "--pid-file", str(pid_file)]


def _read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _wait_until(condition: callable, timeout: int, interval: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False
