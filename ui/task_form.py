"""Render the environment / objective / PPO settings form."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..configs import config_get, config_path


@dataclass
class TaskFormValues:
    """All user-provided inputs for a single generation request."""

    env_path: Path
    objective: str
    run_rl: bool
    max_iterations: int
    max_rl_iterations: int | None
    output_dir: Path
    agent_reasoning: dict[str, bool]
    ppo_kwargs: dict


def render_task_form(st, env_files: list[Path], project_root: Path) -> TaskFormValues:
    """Render the generation form and return the collected values."""

    st.header("Environment")
    labels = [str(p.relative_to(project_root)) for p in env_files]
    default_label = str(config_path("paths.default_env").relative_to(project_root))
    selected = st.selectbox(
        "Environment file",
        labels,
        index=labels.index(default_label) if default_label in labels else 0,
    )
    custom = st.text_input("Custom env path", value=selected)
    path = Path(custom).expanduser()
    env_path = path if path.is_absolute() else project_root / path

    objective = st.text_area(
        "User Prompt",
        value=config_get("ui.objective"),
        height=90,
    )
    run_rl = st.checkbox("Run RL training", value=config_get("ppo.run_rl_training"))
    max_iterations = st.slider(
        "Max reviewer retries" if run_rl else "Max iterations",
        min_value=1,
        max_value=20 if run_rl else 8,
        value=config_get("ui.max_rl_reviewer_retries") if run_rl else config_get("ui.max_static_iterations"),
    )
    max_rl_iterations: int | None = None
    if run_rl:
        max_rl_iterations = st.slider(
            "Max RL iterations",
            min_value=1,
            max_value=8,
            value=config_get("ui.max_rl_iterations"),
            help="Number of PPO runs to start unless the RL Critic accepts earlier.",
        )
    output_dir = Path(st.text_input("Output directory", value=str(config_path("paths.runs_dir"))))

    agent_reasoning = _agent_reasoning_expander(st)

    ppo_kwargs: dict = {}
    if run_rl:
        st.subheader("PPO")
        ppo_kwargs = {
            "num_envs": int(st.number_input("PPO num_envs", min_value=1, value=config_get("ppo.num_envs") or 1, step=1)),
            "max_iterations": int(st.number_input("PPO max_iterations", min_value=10, value=config_get("ppo.max_iterations") or 10, step=10)),
            "timeout_seconds": int(st.number_input("PPO timeout_seconds", min_value=30, value=config_get("ppo.timeout_seconds") or 30, step=30)),
            "headless": st.checkbox("PPO headless", value=config_get("ppo.headless")),
        }

    return TaskFormValues(
        env_path=env_path,
        objective=objective,
        run_rl=run_rl,
        max_iterations=max_iterations,
        max_rl_iterations=max_rl_iterations,
        output_dir=output_dir,
        agent_reasoning=agent_reasoning,
        ppo_kwargs=ppo_kwargs,
    )


def discover_env_files(project_root: Path) -> list[Path]:
    env_dir = project_root / "environments"
    if not env_dir.is_dir():
        return []
    return sorted(p for p in env_dir.glob("*.py") if p.is_file())


def _agent_reasoning_expander(st) -> dict[str, bool]:
    with st.expander("Agent reasoning"):
        st.caption(
            "Thinking mode can improve analysis, but increases output length. "
        )
        return {
            "planner": st.checkbox("Planner reasoning", value=False),
            "coder": st.checkbox("Coder reasoning", value=False),
            "reviewer": st.checkbox("Reviewer reasoning", value=False),
            "critic": st.checkbox("RL Critic reasoning", value=False),
        }
