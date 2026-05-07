"""Render the Streamlit sidebar: server settings, start/stop, log tail."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from ..server import QwenServerSettings, qwen_server_is_ready
from .server_control import (
    PROJECT_ROOT,
    SERVER_LOG_FILE,
    SERVER_PID_FILE,
    build_server_command,
    launch_in_background,
    read_log_tail,
    stop_server,
)


def render_sidebar(st, default_settings: QwenServerSettings) -> tuple[QwenServerSettings, str, list[str]]:
    """Return ``(settings, base_url, server_command)`` after rendering the sidebar."""

    with st.sidebar:
        st.header("Server")
        settings = _settings_inputs(st, default_settings)
        base_url = settings.base_url
        os.environ["QWEN_BASE_URL"] = base_url
        os.environ["QWEN_MODEL"] = settings.served_model_name

        _reachability_indicator(st, base_url)

        server_command = build_server_command(settings)
        with st.expander("Server command"):
            st.code(shlex.join(server_command), language="bash")

        _server_control_buttons(st, settings, server_command)

        log_tail = read_log_tail()
        if log_tail:
            with st.expander("Server log tail"):
                st.code(log_tail, language="text")

    return settings, base_url, server_command


def _settings_inputs(st, default_settings: QwenServerSettings) -> QwenServerSettings:
    host = st.text_input("Host", value=default_settings.host)
    port = st.number_input("Port", min_value=1, max_value=65535, value=default_settings.port, step=1)
    model_path_str = st.text_input("Model path", value=str(default_settings.model_path))
    served_model_name = st.text_input("Served model name", value=default_settings.served_model_name)
    max_model_len = st.number_input("Max model length", min_value=512, value=default_settings.max_model_len, step=512)
    gpu_mem = st.number_input(
        "GPU memory utilization",
        min_value=0.1, max_value=1.0,
        value=default_settings.gpu_memory_utilization,
        step=0.05,
    )
    max_num_seqs = st.number_input("Max num seqs", min_value=1, value=default_settings.max_num_seqs, step=1)

    mp = Path(model_path_str)
    return QwenServerSettings(
        project_root=PROJECT_ROOT,
        model_path=mp if mp.is_absolute() else (PROJECT_ROOT / mp).expanduser(),
        served_model_name=served_model_name,
        reasoning_parser=default_settings.reasoning_parser,
        reasoning_config=default_settings.reasoning_config,
        host=host,
        port=int(port),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_mem),
        max_num_seqs=int(max_num_seqs),
        startup_timeout_seconds=default_settings.startup_timeout_seconds,
        pid_file=default_settings.pid_file,
    )


def _reachability_indicator(st, base_url: str) -> None:
    if qwen_server_is_ready(base_url, timeout=1.0):
        st.success(f"Server reachable: {base_url}")
    else:
        st.warning(f"Server not reachable: {base_url}")


def _server_control_buttons(st, settings: QwenServerSettings, server_command: list[str]) -> None:
    if st.button("Start server", use_container_width=True):
        try:
            launch_in_background(server_command, log_file=SERVER_LOG_FILE)
            st.success("Server startet")
        except Exception as exc:
            st.error(str(exc))

    if st.button("Stop server", use_container_width=True):
        try:
            stopped_pid = stop_server()
            if stopped_pid is None:
                st.warning("No server PID file found.")
            else:
                st.success(f"Sent SIGTERM to server pid={stopped_pid}")
        except ProcessLookupError:
            SERVER_PID_FILE.unlink(missing_ok=True)
            st.warning("Server PID was stale; removed PID file.")
        except Exception as exc:
            st.error(str(exc))
