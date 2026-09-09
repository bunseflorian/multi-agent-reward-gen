"""Kick off a multi-agent run from the UI and render the result."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..configs import build_ppo_config, config_get, config_path
from ..core import RunHooks, RunState, RunStatus, TaskBrief
from ..llm import QwenClient, load_qwen_config
from ..pipeline import RunRecorder, build_graph, build_initial_state, run_dir_name
from ..pipeline.env_context import load_env_context
from ..server import start_server, stop_server
from .task_form import TaskFormValues
from .tensorboard_view import render_tensorboard_metrics

# ---------------------------------------------------------------------------
# Agent panel constants
# ---------------------------------------------------------------------------

_STAGE_ICON = {
    "planner": "🗺️",
    "coder": "💻",
    "reviewer": "🔍",
    "tester": "🧪",
    "ppo": "🏋️",
    "critic": "🎯",
}
_STAGE_LABEL = {
    "planner": "Planner",
    "coder": "Coder",
    "reviewer": "Reviewer",
    "tester": "Tester",
    "ppo": "PPO Training",
    "critic": "Critic",
}


@dataclass
class _UiSlots:
    code_box: Any
    progress_box: Any
    log_box: Any
    tensorboard_box: Any
    agent_panel: Any


def _render_agent_panel(
    placeholder,
    log_entries: list[str],
    ppo_is_active: bool,
    result_status: str | None = None,
) -> None:
    rows = []
    for name in log_entries:
        icon = _STAGE_ICON.get(name, "🤖")
        label = _STAGE_LABEL.get(name, name.capitalize())
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0">'
            f'<span>✅</span><span>{icon}</span><span>{label}</span>'
            f'</div>'
        )

    if ppo_is_active:
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0">'
            f'<span>⏳</span><span>🏋️</span>'
            f'<span style="font-weight:bold">PPO Training</span>'
            f'</div>'
        )

    body = "".join(rows) if rows else (
        '<div style="color:#888;font-size:0.85em;font-style:italic">Waiting for agents…</div>'
    )

    result_html = ""
    if result_status:
        is_failure = result_status == "error" or result_status.startswith("failed")
        icon = "❌" if is_failure else "✅"
        result_html = (
            '<hr style="margin:10px 0">'
            f'<div style="font-size:0.85em">{icon} <code>{result_status}</code></div>'
        )

    placeholder.markdown(
        '<div style="padding:0.5rem 0.75rem;border-left:3px solid #555">'
        '<div style="font-weight:bold;margin-bottom:8px">🤖 Agent Progress</div>'
        + body + result_html + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_generation(st, *, base_url: str, form: TaskFormValues) -> None:
    """Run the full multi-agent pipeline and render the result."""
    if form.build_dpo_pair and form.dpo_target_pairs > 1:
        _collect_dpo_pairs(st, base_url=base_url, form=form)
        return

    col_main, col_agents = st.columns([3, 1])
    slots = _UiSlots(
        code_box=col_main.empty(),
        progress_box=col_main.empty(),
        log_box=col_main.empty(),
        tensorboard_box=col_main.empty(),
        agent_panel=col_agents.empty(),
    )

    pair = _run_single(st, base_url=base_url, form=form, slots=slots)

    if form.build_dpo_pair:
        if pair is not None:
            _write_pair(pair)
            st.success("DPO pair written (1 pair added).")
        else:
            st.info("DPO pair: no pair built (chosen or rejected missing).")


# ---------------------------------------------------------------------------
# DPO collection loop
# ---------------------------------------------------------------------------

def _collect_dpo_pairs(st, *, base_url: str, form: TaskFormValues) -> None:
    progress_hdr = st.empty()
    run_slot = st.empty()

    pairs_collected = 0
    run_idx = 0

    while pairs_collected < form.dpo_target_pairs:
        run_idx += 1
        progress_hdr.info(
            f"⏳ Run {run_idx} · {pairs_collected} / {form.dpo_target_pairs} pairs collected …"
        )

        # Each call to run_slot.container() replaces the previous run's UI.
        ctx = run_slot.container()
        col_main, col_agents = ctx.columns([3, 1])
        slots = _UiSlots(
            code_box=col_main.empty(),
            progress_box=col_main.empty(),
            log_box=col_main.empty(),
            tensorboard_box=col_main.empty(),
            agent_panel=col_agents.empty(),
        )

        pair = _run_single(ctx, base_url=base_url, form=form, slots=slots)
        if pair is not None:
            _write_pair(pair)
            pairs_collected += 1

    progress_hdr.success(
        f"✅ {pairs_collected} / {form.dpo_target_pairs} pairs collected · {run_idx} runs"
    )


# ---------------------------------------------------------------------------
# Single pipeline run
# ---------------------------------------------------------------------------

def _run_single(
    st,
    *,
    base_url: str,
    form: TaskFormValues,
    slots: _UiSlots,
) -> dict | None:
    """Run the pipeline once. Returns a DPO pair dict if one can be built, else None."""
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_run_dir = form.output_dir / f"_running_{form.env_path.stem}_{_ts}"
    tmp_run_dir.mkdir(parents=True)

    log_entries: list[str] = []
    ppo_state = {"active": False}

    def _advance_to(node_name: str) -> None:
        if node_name in ("__end__", "finalizer"):
            return
        if node_name == "tester" and form.run_rl:
            return
        log_entries.append(node_name)
        _render_agent_panel(slots.agent_panel, log_entries, ppo_state["active"])

    ppo_event_log: list[str] = []

    def show_ppo_progress(event: dict) -> None:
        status = event.get("status", "running")
        if status in ("running", "server_stopping", "server_stopped", "server_starting", "server_ready"):
            if not ppo_state["active"] and "ppo" not in log_entries:
                ppo_state["active"] = True
                _render_agent_panel(slots.agent_panel, log_entries, ppo_state["active"])
        elif status in ("completed", "timeout"):
            if ppo_state["active"]:
                ppo_state["active"] = False
                log_entries.append("ppo")
            _render_agent_panel(slots.agent_panel, log_entries, ppo_state["active"])

        label = str(event.get("message")) if event.get("message") else f"PPO {status}"
        elapsed = event.get("elapsed_seconds")
        if elapsed is not None:
            label += f" after {elapsed}s"
        log_path = event.get("log_path")
        if log_path:
            label += f" | {log_path}"
        ppo_event_log.append(label)
        slots.progress_box.info(label)
        tail = event.get("tail") or ""
        slots.log_box.code(tail if tail else "\n".join(ppo_event_log[-20:]), language="text")
        render_tensorboard_metrics(slots.tensorboard_box.container(), tmp_run_dir, live_event=event)

    pre_cb, post_cb = _server_ppo_hooks(base_url, show_ppo_progress) if form.run_rl else (None, None)

    _render_agent_panel(slots.agent_panel, log_entries, ppo_state["active"])

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
            record_video=form.ppo_kwargs.get("record_video"),
            record_gif=form.ppo_kwargs.get("record_gif"),
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
            _advance_to(node_name)
            if isinstance(node_updates, dict):
                accumulated_state.update(node_updates)
                code = accumulated_state.get("code")
                if code is not None:
                    slots.code_box.code(code.code, language="python")

        state = _state_dict_to_run_state(accumulated_state, task_brief=task_brief)

        if ppo_state["active"]:
            ppo_state["active"] = False
            log_entries.append("ppo")

        final_run_dir = form.output_dir / run_dir_name(state)
        state.recorder.move_to(final_run_dir)

        _render_agent_panel(slots.agent_panel, log_entries, False, result_status=state.status.value)
        _render_result(st, state=state, run_dir=final_run_dir, run_rl=form.run_rl)

        if not form.build_dpo_pair:
            return None

        from ..dpo.builder import build_pair
        return build_pair(state)

    except Exception as exc:
        if ppo_state["active"]:
            ppo_state["active"] = False
            log_entries.append("ppo")
        _render_agent_panel(slots.agent_panel, log_entries, False, result_status="error")
        _rename_on_error(tmp_run_dir, form.env_path.stem)
        st.error(str(exc))
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pair(pair: dict) -> None:
    from ..dpo.storage import append_pair
    dpo_path = Path(__file__).parent.parent / "dpo" / "dpo_dataset.jsonl"
    append_pair(pair, dpo_path)


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


def _render_result(st, *, state: RunState, run_dir: Path, run_rl: bool) -> None:
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
        _render_episode_media(st, run_dir)


def _render_episode_media(st, run_dir: Path) -> None:
    iteration_dirs = sorted(p for p in run_dir.glob("rl_run_*") if p.is_dir())
    media_found = False
    for iteration_dir in iteration_dirs:
        gif_path = iteration_dir / "episode.gif"
        video_path = iteration_dir / "episode.mp4"
        if not gif_path.is_file() and not video_path.is_file():
            continue
        if not media_found:
            st.subheader("Episodes")
            media_found = True
        st.caption(iteration_dir.name)
        if gif_path.is_file():
            st.image(str(gif_path))
        elif video_path.is_file():
            st.video(str(video_path))


def _write_error_lines(st, errors: list[str]) -> None:
    for error in errors:
        st.write(error)


def _server_ppo_hooks(base_url: str, progress: callable) -> tuple[callable | None, callable | None]:
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
