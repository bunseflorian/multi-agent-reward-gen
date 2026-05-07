"""Streamlit application entry point.

Run with:
    streamlit run -m qwen_agentv2.ui.app
or::
    streamlit run <path-to>/qwen_agentv2/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is importable when Streamlit invokes this file directly.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from qwen_agentv2.server import load_qwen_server_settings, qwen_server_is_ready  # noqa: E402
from qwen_agentv2.ui.generation import render_pending_plan, run_generation  # noqa: E402
from qwen_agentv2.ui.server_control import PROJECT_ROOT  # noqa: E402
from qwen_agentv2.ui.sidebar import render_sidebar  # noqa: E402
from qwen_agentv2.ui.task_form import discover_env_files, render_task_form  # noqa: E402


def main() -> None:
    """Render the Streamlit application."""

    import streamlit as st

    st.set_page_config(page_title="Qwen Reward UI", layout="wide")
    st.title("Qwen Reward Generator")

    default_settings = load_qwen_server_settings(PROJECT_ROOT)
    settings, base_url, server_command = render_sidebar(st, default_settings)

    env_files = discover_env_files(PROJECT_ROOT)
    if not env_files:
        st.error("No environment files found in ./environments.")
        return

    form = render_task_form(st, env_files, PROJECT_ROOT)

    if render_pending_plan(
        st,
        base_url=base_url,
        server_command=server_command,
        default_settings=default_settings,
    ):
        return

    if not st.button("Generate reward code", type="primary", use_container_width=True):
        return

    if not form.env_path.is_file():
        st.error(f"Environment file does not exist: {form.env_path}")
        return
    if not qwen_server_is_ready(base_url, timeout=2.0):
        st.error(f"Qwen server is not reachable at {base_url}. Start it first.")
        return

    run_generation(
        st,
        base_url=base_url,
        server_command=server_command,
        default_settings=default_settings,
        form=form,
    )


if __name__ == "__main__":
    main()
