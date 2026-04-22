"""Kick off a multi-agent run from the UI and render the result."""

from __future__ import annotations

from pathlib import Path

from ..cli.run import run_task
from ..core import PPOConfig, RunStatus
from ..pipeline import run_dir_name, save_run_state
from ..server import QwenServerSettings
from .server_control import (
    SERVER_LOG_FILE,
    launch_in_background,
    stop_server,
    wait_for_server,
)
from .task_form import TaskFormValues
from .tensorboard_view import render_tensorboard_metrics


def run_generation(
    st,
    *,
    base_url: str,
    server_command: list[str],
    default_settings: QwenServerSettings,
    form: TaskFormValues,
) -> None:
    """Run the multi-agent pipeline and render the outcome in Streamlit."""

    progress_box = st.empty()
    log_box = st.empty()
    tensorboard_box = st.empty()
    ppo_event_log: list[str] = []

    def show_ppo_progress(event: dict) -> None:
        status = event.get("status", "running")
        label = str(event.get("message")) if event.get("message") else f"PPO {status}"
        elapsed = event.get("elapsed_seconds")
        if elapsed is not None:
            label += f" after {elapsed}s"
        log_path = event.get("log_path")
        if log_path:
            label += f" | {log_path}"
        ppo_event_log.append(label)
        progress_box.info(label)
        tail = event.get("tail") or ""
        log_box.code(tail if tail else "\n".join(ppo_event_log[-20:]), language="text")
        with tensorboard_box.container():
            render_tensorboard_metrics(st, tmp_run_dir, live_event=event)

    def pause_qwen() -> None:
        show_ppo_progress({"status": "qwen_stopping", "message": "Stopping Qwen server before PPO to free GPU memory"})
        stopped_pid = stop_server()
        if stopped_pid is None:
            raise RuntimeError("Cannot stop Qwen server before PPO: no server PID file found.")
        show_ppo_progress({"status": "qwen_stopping", "message": f"Sent SIGTERM to Qwen server pid={stopped_pid}"})
        if not wait_for_server(base_url, 60, ready=False):
            raise RuntimeError("Qwen server did not stop before PPO within 60s.")
        show_ppo_progress({"status": "qwen_stopped", "message": "Qwen server stopped; PPO can use GPU memory"})

    def resume_qwen() -> None:
        show_ppo_progress({"status": "qwen_starting", "message": "Restarting Qwen server after PPO for RL Critic"})
        launch_in_background(server_command, log_file=SERVER_LOG_FILE)
        if not wait_for_server(base_url, default_settings.startup_timeout_seconds, ready=True):
            raise RuntimeError("Qwen server did not become ready after PPO.")
        show_ppo_progress({"status": "qwen_ready", "message": f"Qwen server ready again at {base_url}"})

    tmp_run_dir = form.output_dir / f"_running_{form.env_path.stem}"
    tmp_run_dir.mkdir(parents=True, exist_ok=True)
    status_placeholder = st.empty()

    def _update_status(agent: str) -> None:
        status_placeholder.info(f"Running: **{agent}**")

    try:
        state = run_task(
            max_iterations=form.max_iterations,
            max_rl_iterations=form.max_rl_iterations,
            env_path=form.env_path,
            objective=form.objective,
            ppo_config=PPOConfig(
                run_rl_training=form.run_rl,
                num_envs=form.ppo_kwargs.get("num_envs"),
                max_iterations=form.ppo_kwargs.get("max_iterations"),
                timeout_seconds=form.ppo_kwargs.get("timeout_seconds"),
                headless=form.ppo_kwargs.get("headless", True),
                progress_callback=show_ppo_progress if form.run_rl else None,
                pre_training_callback=pause_qwen if form.run_rl else None,
                post_training_callback=resume_qwen if form.run_rl else None,
            ),
            run_dir=tmp_run_dir,
            agent_reasoning=form.agent_reasoning,
            status_callback=_update_status,
        )
        status_placeholder.empty()
        final_run_dir = form.output_dir / run_dir_name(state)
        tmp_run_dir.rename(final_run_dir)
        save_run_state(state, run_dir=final_run_dir)
        run_dir = final_run_dir
    except Exception as exc:
        status_placeholder.empty()
        st.error(str(exc))
        return

    _render_result(st, state=state, run_dir=run_dir, run_rl=form.run_rl)


def _render_result(st, *, state, run_dir: Path, run_rl: bool) -> None:
    st.subheader("Result")
    st.write(f"Status: `{state.status.value}`")
    st.write(f"RL status: `{'skipped' if state.rl_training_skipped else 'enabled'}`")
    st.write(f"Run directory: `{run_dir}`")

    if state.final_report is not None and state.final_report.errors:
        if state.status == RunStatus.FAILED_PLANNING:
            st.error("Planner error")
            _write_error_lines(st, state.final_report.errors)
        elif run_rl and state.status == RunStatus.FAILED_VALIDATION:
            st.error(state.final_report.errors[0])
            if len(state.final_report.errors) > 1:
                _write_error_lines(st, state.final_report.errors[1:])
        else:
            st.error("Static test errors")
            _write_error_lines(st, state.final_report.errors)

    if state.final_code is not None:
        st.code(state.final_code.code, language="python")

    if run_rl:
        render_tensorboard_metrics(st, run_dir)


def _write_error_lines(st, errors: list[str]) -> None:
    for error in errors:
        st.write(error)
