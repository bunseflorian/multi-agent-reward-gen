"""Launch a local vLLM OpenAI-compatible Qwen server with health polling."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import fields
from pathlib import Path

from .settings import QwenServerSettings, load_qwen_server_settings


LOGGER = logging.getLogger("qwen_server")


def start_qwen_server(settings: QwenServerSettings | None = None) -> int:
    """Start a local vLLM server and wait until it responds on ``/v1/models``."""

    _configure_logging()
    settings = settings or load_qwen_server_settings()
    _validate_environment(settings)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = _build_vllm_command(settings)

    LOGGER.info("Starting Qwen server: model=%s address=%s", settings.model_path, settings.base_url)
    process = subprocess.Popen(command, env=env, start_new_session=True)

    if settings.pid_file is not None:
        settings.pid_file.parent.mkdir(parents=True, exist_ok=True)
        settings.pid_file.write_text(str(process.pid), encoding="utf-8")

    def stop_process(signum: int, _frame: object) -> None:
        LOGGER.info("Stopping Qwen server pid=%s after signal %s", process.pid, signum)
        _terminate_process_group(process)

    prev_sigint = signal.signal(signal.SIGINT, stop_process)
    prev_sigterm = signal.signal(signal.SIGTERM, stop_process)
    try:
        LOGGER.info("vLLM pid=%s; waiting for /v1/models", process.pid)
        _wait_for_ready(settings, process)
        LOGGER.info("Qwen server ready at %s (model: %s)", settings.base_url, settings.served_model_name)
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        if process.poll() is None:
            _terminate_process_group(process)
        if settings.pid_file is not None:
            settings.pid_file.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m qwen_agentv2.server.launcher``."""

    parser = argparse.ArgumentParser(description="Start the local Qwen vLLM server.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--reasoning-parser", default=None)
    parser.add_argument("--reasoning-config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    args = parser.parse_args(argv)

    defaults = load_qwen_server_settings()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    merged = {f.name: getattr(defaults, f.name) for f in fields(defaults)}
    merged.update(overrides)

    try:
        return start_qwen_server(QwenServerSettings(**merged))
    except Exception as exc:
        _configure_logging()
        LOGGER.error("%s", exc)
        return 1


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_vllm_command(settings: QwenServerSettings) -> list[str]:
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(settings.model_path),
        "--served-model-name", settings.served_model_name,
        "--reasoning-parser", settings.reasoning_parser,
        "--reasoning-config", settings.reasoning_config,
        "--host", settings.host,
        "--port", str(settings.port),
        "--max-model-len", str(settings.max_model_len),
        "--gpu-memory-utilization", str(settings.gpu_memory_utilization),
        "--max-num-seqs", str(settings.max_num_seqs),
    ]


def _wait_for_ready(settings: QwenServerSettings, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + settings.startup_timeout_seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError("vLLM process exited before the API became ready.")
        try:
            with urllib.request.urlopen(f"{settings.base_url}/models", timeout=2) as resp:
                if resp.status < 400:
                    return
        except Exception:
            pass
        if time.monotonic() >= deadline:
            _terminate_process_group(process)
            raise TimeoutError(f"Timed out waiting for Qwen server at {settings.base_url}/models")
        time.sleep(2)


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()
    process.wait(timeout=10)


def _validate_environment(settings: QwenServerSettings) -> None:
    missing = [name for name in ("torch", "vllm") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing packages: {', '.join(missing)}\n"
            "Install vLLM or activate the server environment: conda activate qwen-vllm-server"
        )
    import torch
    LOGGER.info("torch=%s cuda=%s devices=%s", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA is not visible. vLLM normally needs a CUDA GPU.")
    if not settings.model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {settings.model_path}")
    if not (settings.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model path is missing config.json: {settings.model_path}")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
