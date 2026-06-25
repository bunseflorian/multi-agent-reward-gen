"""Run PPO validation against a patched candidate environment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from ..configs import config_path
from ..core import PPOConfig, PPOReport, TaskBrief
from .command import QWEN_AGENT_EXPERIMENT_NAME, build_ppo_command
from .summary import summarize_ppo_log


def run_ppo_validation(
    task_brief: TaskBrief,
    reward_code: str,
    run_dir: str | Path,
    config: PPOConfig,
    isaaclab_root: Path | None = None,
) -> PPOReport:
    """Patch the candidate env, launch PPO, and return a structured report."""

    root = Path(isaaclab_root) if isaaclab_root is not None else config_path("paths.isaaclab_root")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    candidate_root = config_path("paths.ppo_candidate_dir")
    candidate_root.mkdir(parents=True, exist_ok=True)
    _clear_candidate_env(candidate_root)
    from ..pipeline.env_patch import write_candidate_env

    candidate_path = write_candidate_env(task_brief.env_source_path, reward_code, candidate_root)

    rl_games_root = root / "logs" / "rl_games"
    rl_games_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = rl_games_root / QWEN_AGENT_EXPERIMENT_NAME

    command = build_ppo_command(
        config,
        isaaclab_root=root,
        rl_games_root=rl_games_root,
        experiment_name=QWEN_AGENT_EXPERIMENT_NAME,
    )
    log_path = run_path / "ppo.log"

    exit_code, timed_out = _run_command_with_progress(command, root, log_path, config)

    summary = summarize_ppo_log(
        log_path,
        exit_code=exit_code,
        timed_out=timed_out,
        experiment_dir=experiment_dir,
        experiment_search_root=rl_games_root,
    )
    summary["candidate_env_path"] = str(candidate_path)
    summary["candidate_env_root"] = str(candidate_root)
    summary["rl_games_root"] = str(rl_games_root)
    summary["candidate_reward_sha256"] = hashlib.sha256(reward_code.encode("utf-8")).hexdigest()

    _save_policy_artifacts(run_path, summary.get("experiment_dir"), candidate_path)

    summary_path = run_path / "ppo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    errors = list(summary.get("errors", []))
    status = "timeout" if timed_out else "passed" if exit_code == 0 else "failed"
    passed = status == "passed" or (timed_out and not errors)
    warnings = (
        ["PPO stopped at the configured timeout; treated as a bounded training stop."]
        if timed_out and passed
        else []
    )
    return PPOReport(
        status=status,
        passed=passed,
        command=command,
        log_path=str(log_path),
        summary_path=str(summary_path),
        exit_code=exit_code,
        timed_out=timed_out,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def skipped_ppo_report() -> PPOReport:
    """Return a passing report used when PPO is disabled."""

    return PPOReport(status="skipped", passed=True)


# ---------------------------------------------------------------------------
# Process supervision
# ---------------------------------------------------------------------------

def _run_command_with_progress(
    command: list[str],
    cwd: Path,
    log_path: Path,
    config: PPOConfig,
) -> tuple[int | None, bool]:
    """Run a command while streaming progress events from the log file."""

    started_at = time.monotonic()
    started_at_wall = time.time()
    experiment_search_root = cwd / "logs" / "rl_games" / QWEN_AGENT_EXPERIMENT_NAME
    with log_path.open("wb") as log_file:
        log_file.write(("Command: " + " ".join(command) + "\n").encode("utf-8"))
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _emit_progress(
            config,
            {
                "status": "running",
                "pid": process.pid,
                "log_path": str(log_path),
                "tail": "",
                "experiment_search_root": str(experiment_search_root),
                "started_at_wall": started_at_wall,
            },
        )

        while True:
            exit_code = process.poll()
            if exit_code is not None:
                _emit_progress(
                    config,
                    {
                        "status": "completed",
                        "exit_code": exit_code,
                        "log_path": str(log_path),
                        "tail": _read_tail(log_path),
                        "experiment_search_root": str(experiment_search_root),
                        "started_at_wall": started_at_wall,
                    },
                )
                return exit_code, False

            elapsed = time.monotonic() - started_at
            if config.timeout_seconds is not None and elapsed >= config.timeout_seconds:
                _terminate_process_group(process)
                _emit_progress(
                    config,
                    {
                        "status": "timeout",
                        "log_path": str(log_path),
                        "tail": _read_tail(log_path),
                        "experiment_search_root": str(experiment_search_root),
                        "started_at_wall": started_at_wall,
                    },
                )
                return None, True

            _emit_progress(
                config,
                {
                    "status": "running",
                    "elapsed_seconds": int(elapsed),
                    "log_path": str(log_path),
                    "tail": _read_tail(log_path),
                    "experiment_search_root": str(experiment_search_root),
                    "started_at_wall": started_at_wall,
                },
            )
            time.sleep(2)


def _emit_progress(config: PPOConfig, payload: dict) -> None:
    if config.hooks.ppo_progress_callback is not None:
        config.hooks.ppo_progress_callback(payload)


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate the launched command and all children in its session."""

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


def _clear_candidate_env(run_path: Path) -> None:
    candidate_dir = run_path / "candidate_env"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)


def _read_tail(path: Path, max_chars: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return text[-max_chars:]


def _save_policy_artifacts(
    run_path: Path,
    experiment_dir: str | None,
    candidate_env_path: Path,
) -> None:
    """Copy policy checkpoints and the candidate env into the run directory."""
    if experiment_dir is not None:
        nn_src = Path(experiment_dir) / "nn"
        if nn_src.is_dir():
            shutil.copytree(nn_src, run_path / "policy", dirs_exist_ok=True)
    shutil.copy2(candidate_env_path, run_path / "env.py")
