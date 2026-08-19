"""Kick off a multi-agent run from the UI and render the result."""

from __future__ import annotations

import urllib.request
from datetime import datetime
from pathlib import Path

from ..configs import build_ppo_config, config_get, config_path
from ..core import PPOConfig, RunHooks, RunState, RunStatus, TaskBrief
from ..llm import QwenClient, load_qwen_config
from ..pipeline import RunRecorder, build_graph, build_initial_state, run_dir_name
from ..pipeline.env_context import load_env_context
from ..server import start_server, stop_server
from .task_form import TaskFormValues
from .tensorboard_view import render_tensorboard_metrics


def run_generation(
    st,
    *,
    base_url: str,
    form: TaskFormValues,
) -> None:
    """Run the full multi-agent pipeline and render the result."""

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_run_dir = form.output_dir / f"_running_{form.env_path.stem}_{_ts}"
    tmp_run_dir.mkdir(parents=True)

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

    pre_cb, post_cb = _server_ppo_hooks(base_url, show_ppo_progress) if form.run_rl else (None, None)
    status_placeholder = st.empty()

    try:
        client = QwenClient(load_qwen_config())

        resolved_env: Path = form.env_path
        if form.randomize_env:
            from ..pipeline.env_randomizer import create_candidate_env
            resolved_env = create_candidate_env(resolved_env, tmp_run_dir)

        env_source_path, env_code_context, _ = load_env_context(
            resolved_env, config_path("paths.qwen_agent_root")
        )
        task_brief = TaskBrief(
            task_id=Path(resolved_env).stem,
            objective=form.objective,
            env_source_path=str(env_source_path),
            env_code_context=env_code_context,
        )

        recorder = RunRecorder(tmp_run_dir)
        ppo_config = build_ppo_config(
            run_rl_training=form.run_rl,
            num_envs=form.ppo_kwargs.get("num_envs"),
            max_iterations=form.ppo_kwargs.get("max_iterations"),
            timeout_seconds=form.ppo_kwargs.get("timeout_seconds"),
            headless=form.ppo_kwargs.get("headless", True),
            hooks=RunHooks(
                ppo_progress_callback=show_ppo_progress if form.run_rl else None,
                pre_training_callback=pre_cb,
                post_training_callback=post_cb,
            ),
        )

        compiled = build_graph(client, recorder=recorder)
        initial_state = build_initial_state(
            task_brief=task_brief,
            ppo_config=ppo_config,
            max_iterations=form.max_iterations,
            max_rl_iterations=form.max_rl_iterations,
            run_dir=str(tmp_run_dir),
            agent_reasoning=form.agent_reasoning,
            recorder=recorder,
        )

        accumulated_state = dict(initial_state)
        for event in compiled.stream(initial_state, stream_mode="updates"):
            node_name = list(event.keys())[0]
            node_updates = event[node_name]
            if node_name != "__end__":
                status_placeholder.info(f"Running: **{node_name}**")
            if isinstance(node_updates, dict):
                accumulated_state.update(node_updates)

        status_placeholder.empty()
        state = _state_dict_to_run_state(accumulated_state, task_brief=task_brief)
        final_run_dir = form.output_dir / run_dir_name(state)
        state.recorder.move_to(final_run_dir)
        if form.build_dpo_pair:
            from ..dpo.builder import build_pair
            from ..dpo.storage import append_pair
            pair = build_pair(state)
            if pair is not None:
                dpo_path = Path(__file__).parent.parent / "dpo" / "dpo_dataset.jsonl"
                append_pair(pair, dpo_path)
                st.success("DPO pair written (1 pair added).")
            else:
                st.info("DPO pair: no pair built (chosen or rejected missing).")
        run_dir = final_run_dir
    except Exception as exc:
        status_placeholder.empty()
        _rename_on_error(tmp_run_dir, form.env_path.stem)
        st.error(str(exc))
        return

    _render_result(st, state=state, run_dir=run_dir, run_rl=form.run_rl)


def _state_dict_to_run_state(state: dict, *, task_brief: TaskBrief) -> RunState:
    run_state = RunState(
        task_brief=task_brief,
        status=RunStatus(state["status"]),
        iterations=state.get("iterations", []),
        plan=state.get("plan"),
        final_code=state.get("final_code"),
        final_report=state.get("final_report"),
        final_ppo_report=state.get("final_ppo_report"),
        rl_training_skipped=not state["ppo_config"].run_rl_training,
    )
    run_state.recorder = state.get("recorder")
    return run_state


def _rename_on_error(tmp_dir: Path, task_stem: str) -> None:
    if not tmp_dir.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_dir = tmp_dir.parent / f"{ts}_{task_stem}_error"
    try:
        tmp_dir.rename(failed_dir)
    except Exception:
        pass


def _is_server_ready(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            return resp.status < 400
    except Exception:
        return False


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


def _server_ppo_hooks(base_url: str, progress: callable) -> tuple[callable | None, callable | None]:
    """Return (pre_training, post_training) callbacks if the server is managed locally."""
    launcher_str = config_get("server.launcher")
    pid_file_str = config_get("server.pid_file")
    timeout = int(config_get("server.startup_timeout_seconds", 300))

    if not launcher_str or not pid_file_str:
        return None, None

    launcher = Path(launcher_str)
    pid_file = Path(pid_file_str)

    def pre_training() -> None:
        progress({"status": "server_stopping", "message": "Killing vLLM server to free GPU memory for PPO"})
        stop_server(pid_file, base_url)
        progress({"status": "server_stopped", "message": "vLLM server killed — starting PPO"})

    def post_training() -> None:
        progress({"status": "server_starting", "message": "Restarting vLLM server after PPO"})
        if not start_server(launcher, pid_file, base_url, timeout=timeout):
            raise RuntimeError(f"vLLM server did not become ready within {timeout}s after PPO.")
        progress({"status": "server_ready", "message": f"vLLM server ready again at {base_url}"})

    return pre_training, post_training
