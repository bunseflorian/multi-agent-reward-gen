"""Render the Streamlit sidebar: server URL, reachability and start button."""

from __future__ import annotations

import os
from pathlib import Path

from ..configs import config_get
from ..server import qwen_server_is_ready, start_server, stop_server


def render_sidebar(st) -> str:
    """Render the sidebar and return the configured server base URL."""

    with st.sidebar:
        st.header("LLM Server")
        default_url = config_get("llm.base_url", "http://localhost:8000/v1")
        base_url = st.text_input("Server URL", value=default_url)
        os.environ["QWEN_BASE_URL"] = base_url

        is_ready = qwen_server_is_ready(base_url, timeout=1.0)
        if is_ready:
            st.success(f"Reachable: {base_url}")
        else:
            st.warning(f"Not reachable: {base_url}")

        launcher_str = config_get("server.launcher")
        pid_file_str = config_get("server.pid_file")
        if launcher_str and pid_file_str:
            timeout = int(config_get("server.startup_timeout_seconds", 300))
            pid_file = Path(pid_file_str)
            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("Start", disabled=is_ready, use_container_width=True):
                    with st.spinner("Starting vLLM server…"):
                        ok = start_server(Path(launcher_str), pid_file, base_url, timeout=timeout)
                    if ok:
                        st.rerun()
                    else:
                        st.error(f"Server did not become ready within {timeout}s.")
            with col_stop:
                if st.button("Stop", disabled=not is_ready, use_container_width=True):
                    with st.spinner("Stopping vLLM server…"):
                        stop_server(pid_file, base_url)
                    st.rerun()

    return base_url
