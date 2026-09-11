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
from ..server import wait_for_gpu_memory_released
from .command import QWEN_AGENT_EXPERIMENT_NAME, build_ppo_command, build_play_command
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

    errors = list(summary.get("errors", []))
    status = "timeout" if timed_out else "passed" if exit_code == 0 else "failed"
    passed = status == "passed" or (timed_out and not errors)
    warnings = (
        ["PPO stopped at the configured timeout; treated as a bounded training stop."]
        if timed_out and passed
        else []
    )

    if (config.record_video or config.record_gif) and passed and summary.get("experiment_dir"):
        video_path, gif_path = _record_episode_video(run_path, Path(summary["experiment_dir"]), config, root)
        if video_path is not None:
            summary["episode_video_path"] = str(video_path)
        if gif_path is not None:
            summary["episode_gif_path"] = str(gif_path)

    summary_path = run_path / "ppo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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


def _find_final_checkpoint(nn_dir: Path) -> Path | None:
    """Return the checkpoint to play; falls back to the newest `last_*` save if the canonical file is missing."""

    canonical = nn_dir / f"{QWEN_AGENT_EXPERIMENT_NAME}.pth"
    if canonical.is_file():
        return canonical
    checkpoints = [p for p in nn_dir.glob("*.pth") if p.is_file()] if nn_dir.is_dir() else []
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def _record_episode_video(
    run_path: Path,
    experiment_dir: Path,
    config: PPOConfig,
    isaaclab_root: Path,
) -> tuple[Path | None, Path | None]:
    """Play the final checkpoint for one episode; returns (mp4_path, gif_path), either possibly None. Best-effort."""

    checkpoint_path = _find_final_checkpoint(experiment_dir / "nn")
    if checkpoint_path is None:
        return None, None

    command = build_play_command(config, checkpoint_path, isaaclab_root=isaaclab_root)
    play_log_path = run_path / "play.log"
    timed_out = False
    try:
        with play_log_path.open("wb") as play_log:
            # conda run doesn't forward signals reliably; kill the whole group on timeout.
            process = subprocess.Popen(
                command,
                cwd=isaaclab_root,
                stdout=play_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=config.video_timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                timed_out = True
    except OSError:
        return None, None
    finally:
        wait_for_gpu_memory_released()  # driver may lag process exit; caller restarts vLLM right after

    if timed_out:
        return None, None

    video_dir = experiment_dir / "videos" / "play"
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        return None, None
    source_mp4 = videos[-1]

    mp4_path: Path | None = None
    if config.record_video:
        mp4_path = run_path / "episode.mp4"
        shutil.copy2(source_mp4, mp4_path)

    gif_path = _convert_to_gif(source_mp4, run_path / "episode.gif") if config.record_gif else None
    return mp4_path, gif_path


def _convert_to_gif(mp4_path: Path, gif_path: Path, fps: int = 10, width: int = 480) -> Path | None:
    """Convert a recorded episode to a palette-optimized GIF via ffmpeg. Best-effort."""

    filter_chain = f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", filter_chain, str(gif_path)],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not gif_path.is_file():
        gif_path.with_suffix(".gif.log").write_bytes(result.stderr)
        return None
    return gif_path


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
