"""Extract redacted environment context for planner and coder prompts."""

from __future__ import annotations

import ast
from pathlib import Path

from ..configs import config_path
from .env_patch import redact_reward_body


# Methods extracted from the env file for the prompt context.
_RELEVANT_METHODS = (
    "_pre_physics_step",
    "_get_observations",
    "_get_rewards",
    "_compute_generated_reward",
    "_get_dones",
    "_reset_idx",
)


def detect_env_classes(env_path: str | Path) -> tuple[str, str]:
    """Auto-detect env and cfg class names from an environment file.

    Returns ``(env_class_name, env_cfg_class_name)``.
    """

    source = Path(env_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    env_class: str | None = None
    cfg_class: str | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.endswith("EnvCfg"):
            cfg_class = node.name
        elif node.name.endswith("Env"):
            env_class = node.name
    if env_class is None:
        raise ValueError(f"No *Env class found in {env_path}")
    if cfg_class is None:
        raise ValueError(f"No *EnvCfg class found in {env_path}")
    return env_class, cfg_class


def load_env_context(
    env_path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[Path, str, tuple[str, str]]:
    """Resolve env path, extract prompt context, and detect class names."""

    root = Path(project_root) if project_root is not None else config_path("paths.qwen_agent_root")
    path = Path(env_path).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Required environment file not found: {path}")
    env_class, cfg_class = detect_env_classes(path)
    context = extract_env_code_context(path, env_class, cfg_class)
    return path, context, (env_class, cfg_class)


def extract_env_code_context(
    env_path: str | Path,
    env_class: str | None = None,
    cfg_class: str | None = None,
) -> str:
    """Extract reward-relevant source sections from an environment file."""

    path = Path(env_path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    if env_class is None or cfg_class is None:
        env_class_auto, cfg_class_auto = detect_env_classes(path)
        env_class = env_class or env_class_auto
        cfg_class = cfg_class or cfg_class_auto

    chunks = list(_extract_chunks(tree, lines, env_class, cfg_class))

    all_sections = (cfg_class,) + _RELEVANT_METHODS
    missing = [name for name in all_sections if name not in "\n".join(chunks)]
    header = [
        f"Source file: {path}",
        "Relevant extracted sections:",
        *(f"- {name}" for name in all_sections),
    ]
    if missing:
        header += ["", "Missing expected sections:", *(f"- {name}" for name in missing)]
    return "\n".join(header) + "\n\n```python\n" + "\n\n".join(chunks).strip() + "\n```"


def _extract_chunks(
    tree: ast.Module,
    lines: list[str],
    env_class: str,
    cfg_class: str,
) -> list[str]:
    chunks: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cfg_class:
            chunks.append(_extract_node_source(lines, node))
        elif isinstance(node, ast.ClassDef) and node.name == env_class:
            method_chunks = [
                _extract_node_source(lines, child)
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name in _RELEVANT_METHODS
            ]
            if method_chunks:
                chunks.append(
                    f"class {env_class}:\n"
                    + "\n\n".join(_indent(chunk, "    ") for chunk in method_chunks)
                )
    return chunks


def _extract_node_source(lines: list[str], node: ast.AST) -> str:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    if isinstance(node, ast.FunctionDef) and node.name == "_compute_generated_reward":
        return redact_reward_body(source)
    return source


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())
