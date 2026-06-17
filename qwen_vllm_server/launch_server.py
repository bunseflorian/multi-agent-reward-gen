"""Standalone vLLM server launcher for Qwen models.

No dependency on IsaacLab or the qwen_agentv2 project.
Requires only: vllm, pyyaml  (see requirements.txt)

Usage:
    python launch_server.py
    python launch_server.py --model-path /data/models/qwen --host 0.0.0.0 --port 8000

Environment variable overrides (take precedence over config.yaml):
    QWEN_MODEL_PATH, QWEN_SERVED_MODEL_NAME, QWEN_HOST, QWEN_PORT,
    QWEN_MAX_MODEL_LEN, QWEN_GPU_MEMORY_UTILIZATION, QWEN_MAX_NUM_SEQS,
    QWEN_STARTUP_TIMEOUT_SECONDS, QWEN_REASONING_PARSER, QWEN_REASONING_CONFIG
"""

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
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("qwen_server")
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerSettings:
    model_path: Path
    served_model_name: str
    host: str
    port: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    startup_timeout_seconds: int
    reasoning_parser: str
    reasoning_config: str
    pid_file: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def load_settings(config_path: Path = CONFIG_PATH) -> ServerSettings:
    """Load settings from config.yaml, then apply env var overrides."""
    cfg: dict[str, Any] = {}
    if config_path.is_file():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg = data

    def _get(key: str, default: Any = None) -> Any:
        return cfg.get(key, default)

    env = os.environ
    raw_model_path = Path(env.get("QWEN_MODEL_PATH") or _get("model_path", "."))
    model_path = raw_model_path if raw_model_path.is_absolute() else (config_path.parent / raw_model_path).resolve()
    return ServerSettings(
        model_path=model_path,
        served_model_name=env.get("QWEN_SERVED_MODEL_NAME") or _get("served_model_name", "Qwen/Qwen3.5-4B"),
        host=env.get("QWEN_HOST") or _get("host", "0.0.0.0"),
        port=int(env.get("QWEN_PORT") or _get("port", 8000)),
        max_model_len=int(env.get("QWEN_MAX_MODEL_LEN") or _get("max_model_len", 11776)),
        gpu_memory_utilization=float(env.get("QWEN_GPU_MEMORY_UTILIZATION") or _get("gpu_memory_utilization", 0.85)),
        max_num_seqs=int(env.get("QWEN_MAX_NUM_SEQS") or _get("max_num_seqs", 1)),
        startup_timeout_seconds=int(env.get("QWEN_STARTUP_TIMEOUT_SECONDS") or _get("startup_timeout_seconds", 300)),
        reasoning_parser=env.get("QWEN_REASONING_PARSER") or _get("reasoning_parser", "qwen3"),
        reasoning_config=env.get("QWEN_REASONING_CONFIG") or _get("reasoning_config", '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'),
    )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(settings: ServerSettings) -> int:
    """Launch vLLM, wait until /v1/models responds, then block until exit."""
    _configure_logging()
    _validate_environment(settings)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = _build_vllm_command(settings)

    LOGGER.info("Starting server: model=%s  address=%s", settings.model_path, settings.base_url)
    process = subprocess.Popen(command, env=env, start_new_session=True)

    if settings.pid_file is not None:
        settings.pid_file.parent.mkdir(parents=True, exist_ok=True)
        settings.pid_file.write_text(str(process.pid), encoding="utf-8")

    def _stop(signum: int, _frame: object) -> None:
        LOGGER.info("Stopping server pid=%s after signal %s", process.pid, signum)
        _terminate_process_group(process)

    prev_sigint = signal.signal(signal.SIGINT, _stop)
    prev_sigterm = signal.signal(signal.SIGTERM, _stop)
    try:
        LOGGER.info("vLLM pid=%s — waiting for /v1/models …", process.pid)
        _wait_for_ready(settings, process)
        LOGGER.info("Server ready at %s  (model: %s)", settings.base_url, settings.served_model_name)
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        if process.poll() is None:
            _terminate_process_group(process)
        if settings.pid_file is not None:
            settings.pid_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_vllm_command(s: ServerSettings) -> list[str]:
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(s.model_path),
        "--served-model-name", s.served_model_name,
        "--reasoning-parser", s.reasoning_parser,
        "--reasoning-config", s.reasoning_config,
        "--host", s.host,
        "--port", str(s.port),
        "--max-model-len", str(s.max_model_len),
        "--gpu-memory-utilization", str(s.gpu_memory_utilization),
        "--max-num-seqs", str(s.max_num_seqs),
    ]


def _wait_for_ready(settings: ServerSettings, process: subprocess.Popen) -> None:
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
            raise TimeoutError(f"Timed out waiting for server at {settings.base_url}/models")
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


def _validate_environment(settings: ServerSettings) -> None:
    missing = [name for name in ("torch", "vllm") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing packages: {', '.join(missing)}\n"
            "Run:  pip install -r requirements.txt"
        )
    import torch
    LOGGER.info("torch=%s  cuda=%s  devices=%s", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA not visible — vLLM needs a CUDA GPU.")
    if not settings.model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {settings.model_path}")
    if not (settings.model_path / "config.json").is_file():
        raise FileNotFoundError(f"config.json missing in model path: {settings.model_path}")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Start the Qwen vLLM server.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    args = parser.parse_args(argv)

    defaults = load_settings(args.config)
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("config",)}
    merged = {f.name: getattr(defaults, f.name) for f in fields(defaults)}
    merged.update(cli_overrides)

    try:
        return start_server(ServerSettings(**merged))
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
