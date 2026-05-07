"""Kick off a multi-agent run from the UI and render the result."""

from __future__ import annotations

from pathlib import Path

from ..configs import build_ppo_config
from ..cli.run import run_task
from ..core import PPOConfig, RunHooks, RunStatus
from ..pipeline import run_dir_name
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
    """Run the Planner and leave the generated plan pending for UI approval."""

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
            ppo_config=_build_ppo_config(form, hooks=RunHooks()),
            run_dir=tmp_run_dir,
            agent_reasoning=form.agent_reasoning,
            status_callback=_update_status,
            stop_after_plan=True,
        )
        status_placeholder.empty()
    except Exception as exc:
        status_placeholder.empty()
        st.error(str(exc))
        return

    st.session_state["pending_plan_run"] = {
        "state": state,
        "form": form,
        "run_dir": tmp_run_dir,
    }
    render_pending_plan(
        st,
        base_url=base_url,
        server_command=server_command,
        default_settings=default_settings,
    )


def render_pending_plan(
    st,
    *,
    base_url: str,
    server_command: list[str],
    default_settings: QwenServerSettings,
) -> bool:
    """Render Accept/Reject controls for a plan created in a previous UI run."""

    pending = st.session_state.get("pending_plan_run")
    if not pending:
        return False

    state = pending["state"]
    plan = state.plan
    if plan is None:
        st.warning("Pending run has no planner output.")
        return True

    st.subheader("Planner Output")
    st.write(f"Run directory: `{pending['run_dir']}`")
    st.code(plan.text, language="text")

    accept_col, reject_col = st.columns(2)
    with accept_col:
        accept = st.button("Accept plan and continue", type="primary", use_container_width=True)
    with reject_col:
        reject = st.button("Reject plan", use_container_width=True)

    if reject:
        _reject_pending_plan(st, pending)
        return True

    if accept:
        if not wait_for_server(base_url, 2, ready=True):
            st.error(f"Qwen server is not reachable at {base_url}. Start it first.")
            return True
        _continue_pending_plan(
            st,
            pending,
            base_url=base_url,
            server_command=server_command,
            default_settings=default_settings,
        )
        return True

    return True


def _continue_pending_plan(
    st,
    pending: dict,
    *,
    base_url: str,
    server_command: list[str],
    default_settings: QwenServerSettings,
) -> None:
    form: TaskFormValues = pending["form"]
    planned_state = pending["state"]
    tmp_run_dir: Path = pending["run_dir"]

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

    def _update_status(agent: str) -> None:
        status_placeholder.info(f"Running: **{agent}**")

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

    status_placeholder = st.empty()

    try:
        state = run_task(
            max_iterations=form.max_iterations,
            max_rl_iterations=form.max_rl_iterations,
            env_path=form.env_path,
            objective=form.objective,
            ppo_config=build_ppo_config(
                run_rl_training=form.run_rl,
                num_envs=form.ppo_kwargs.get("num_envs"),
                max_iterations=form.ppo_kwargs.get("max_iterations"),
                timeout_seconds=form.ppo_kwargs.get("timeout_seconds"),
                headless=form.ppo_kwargs.get("headless", True),
                hooks=RunHooks(
                    ppo_progress_callback=show_ppo_progress if form.run_rl else None,
                    pre_training_callback=pause_qwen if form.run_rl else None,
                    post_training_callback=resume_qwen if form.run_rl else None,
                ),
            ),
            run_dir=tmp_run_dir,
            agent_reasoning=form.agent_reasoning,
            status_callback=_update_status,
            initial_plan=planned_state.plan,
        )
        status_placeholder.empty()
        final_run_dir = form.output_dir / run_dir_name(state)
        tmp_run_dir.rename(final_run_dir)
        state.recorder.move_to(final_run_dir)
        state.recorder.save_run_state(state)
        run_dir = final_run_dir
        st.session_state.pop("pending_plan_run", None)
    except Exception as exc:
        status_placeholder.empty()
        st.error(str(exc))
        return

    _render_result(st, state=state, run_dir=run_dir, run_rl=form.run_rl)


def _reject_pending_plan(st, pending: dict) -> None:
    state = pending["state"]
    tmp_run_dir: Path = pending["run_dir"]
    form: TaskFormValues = pending["form"]

    state.status = RunStatus.REJECTED_PLAN
    if state.recorder is not None:
        state.recorder.event("plan_rejected")
        state.recorder.event("run_finished", status=state.status.value, accepted=state.accepted)
        state.recorder.save_run_state(state)

    final_run_dir = form.output_dir / run_dir_name(state)
    tmp_run_dir.rename(final_run_dir)
    if state.recorder is not None:
        state.recorder.move_to(final_run_dir)
        state.recorder.save_run_state(state)

    st.session_state.pop("pending_plan_run", None)
    st.info(f"Plan rejected. Run saved to `{final_run_dir}`.")


def _build_ppo_config(form: TaskFormValues, *, hooks: RunHooks) -> PPOConfig:
    return build_ppo_config(
        run_rl_training=form.run_rl,
        num_envs=form.ppo_kwargs.get("num_envs"),
        max_iterations=form.ppo_kwargs.get("max_iterations"),
        timeout_seconds=form.ppo_kwargs.get("timeout_seconds"),
        headless=form.ppo_kwargs.get("headless", True),
        hooks=hooks,
    )


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
