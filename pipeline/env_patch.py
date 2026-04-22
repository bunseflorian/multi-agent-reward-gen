"""Redact and patch reward bodies in IsaacLab environment files."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def redact_reward_body(source: str) -> str:
    """Return source with the generated reward body replaced by a placeholder."""

    dedented = textwrap.dedent(source)
    leading_indent = source[: len(source) - len(source.lstrip(" "))]
    tree = ast.parse(dedented)
    if tree.body and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "_compute_generated_reward":
        signature = dedented.splitlines()[0]
        replacement = [signature, "    # Reward body redacted for planning.", "    pass"]
        return textwrap.indent("\n".join(replacement), leading_indent)

    lines = source.splitlines()
    node = _find_method(tree, "_compute_generated_reward") or _find_method(tree, "_get_rewards")
    if node is None or node.end_lineno is None:
        return source
    indent = " " * (node.col_offset + 4)
    replacement = [
        lines[node.lineno - 1],
        f"{indent}# Reward body redacted for planning.",
        f"{indent}pass",
    ]
    return "\n".join(lines[: node.lineno - 1] + replacement + lines[node.end_lineno:])


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_reward_body(source: str, reward_code: str) -> str:
    """Replace ``_compute_generated_reward`` body with generated reward code."""

    source = ensure_generated_reward_helper(source)
    tree = ast.parse(source)
    lines = source.splitlines()
    node = _find_method(tree, "_compute_generated_reward")
    if node is None or node.end_lineno is None:
        raise ValueError("Could not find _compute_generated_reward method to patch.")

    indent = " " * (node.col_offset + 4)
    reward_lines = reward_code.strip().splitlines()
    if not reward_lines:
        raise ValueError("Reward code is empty.")

    body = [
        f"{indent}if not hasattr(self, '_episode_sums'):",
        f"{indent}    self._episode_sums = {{}}",
        f"{indent}if not hasattr(self, '_episode_final_values'):",
        f"{indent}    self._episode_final_values = {{}}",
        f"{indent}if not hasattr(self, '_episode_max_values'):",
        f"{indent}    self._episode_max_values = {{}}",
        "",
        *[f"{indent}{line}" if line.strip() else "" for line in reward_lines],
        "",
        f"{indent}return reward",
    ]
    return "\n".join(lines[: node.lineno] + body + lines[node.end_lineno:]) + "\n"


def ensure_generated_reward_helper(source: str) -> str:
    """Ensure ``_get_rewards`` delegates to ``_compute_generated_reward``."""

    tree = ast.parse(source)
    if _find_method(tree, "_compute_generated_reward") is not None:
        return source

    node = _find_method(tree, "_get_rewards")
    if node is None or node.end_lineno is None:
        raise ValueError("Could not find _get_rewards method to patch.")

    lines = source.splitlines()
    indent = " " * (node.col_offset + 4)
    method_indent = " " * node.col_offset
    replacement = [
        lines[node.lineno - 1],
        f"{indent}return self._compute_generated_reward()",
        "",
        f"{method_indent}def _compute_generated_reward(self) -> torch.Tensor:",
        f"{indent}# QWEN-GENERATED REWARD FUNCTION",
        "",
        f"{indent}return reward",
    ]
    return "\n".join(lines[: node.lineno - 1] + replacement + lines[node.end_lineno:]) + "\n"


def patch_reset_idx_for_generated_metrics(source: str) -> str:
    """Inject generated-metric logging into ``_reset_idx``.

    The block is inserted just before the ``super()._reset_idx(...)`` call so
    dynamically-named metrics written by the Coder reach TensorBoard.
    """

    tree = ast.parse(source)
    reset_node = _find_reset_idx(tree)
    if reset_node is None or reset_node.end_lineno is None:
        return source

    if _method_references_generated_metrics(reset_node):
        return source

    indent = " " * (reset_node.col_offset + 4)
    block = _build_generated_metrics_block(indent)

    lines = source.splitlines()
    insert_line = reset_node.end_lineno
    for stmt in reset_node.body:
        if _is_super_reset_idx_call(stmt):
            insert_line = stmt.lineno - 1
            break

    return "\n".join(lines[:insert_line] + block + lines[insert_line:]) + "\n"


# ---------------------------------------------------------------------------
# Candidate env writer
# ---------------------------------------------------------------------------

def write_candidate_env(env_path: str | Path, reward_code: str, run_dir: str | Path) -> Path:
    """Write a patched candidate env as ``candidate_env/candidate_env.py``.

    Also writes ``__init__.py`` and ``manifest.json`` so the parent package can
    register the gym task without knowing the original filename.
    """

    source_path = Path(env_path)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    candidate_dir = Path(run_dir) / "candidate_env"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    patched = patch_reward_body(source, reward_code)
    patched = patch_reset_idx_for_generated_metrics(patched)

    candidate_path = candidate_dir / "candidate_env.py"
    candidate_path.write_text(patched, encoding="utf-8")
    (candidate_dir / "__init__.py").write_text("", encoding="utf-8")

    manifest = {
        "env_class": _detect_env_class(tree) or "UnknownEnv",
        "env_cfg_class": _detect_cfg_class(tree) or "UnknownEnvCfg",
    }
    (candidate_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return candidate_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_generated_metrics_block(indent: str) -> list[str]:
    """Return the generic generated-metric logging block for ``_reset_idx``."""

    mean_block = [
        f"{indent}if hasattr(self, '_episode_sums'):",
        f"{indent}    if 'log' not in self.extras:",
        f"{indent}        self.extras['log'] = {{}}",
        f"{indent}    _n_steps = self.episode_length_buf[env_ids].float().clamp(min=1.0)",
        f"{indent}    for _key, _values in self._episode_sums.items():",
        f"{indent}        if isinstance(_values, torch.Tensor):",
        f"{indent}            self.extras['log'][_key] = (_values[env_ids] / _n_steps).mean().item()",
        f"{indent}            self._episode_sums[_key][env_ids] = 0.0",
    ]
    final_block = [
        f"{indent}if hasattr(self, '_episode_final_values'):",
        f"{indent}    if 'log' not in self.extras:",
        f"{indent}        self.extras['log'] = {{}}",
        f"{indent}    for _key, _values in self._episode_final_values.items():",
        f"{indent}        if isinstance(_values, torch.Tensor):",
        f"{indent}            self.extras['log'][_key] = _values[env_ids].mean().item()",
        f"{indent}            self._episode_final_values[_key][env_ids] = 0.0",
    ]
    max_block = [
        f"{indent}if hasattr(self, '_episode_max_values'):",
        f"{indent}    if 'log' not in self.extras:",
        f"{indent}        self.extras['log'] = {{}}",
        f"{indent}    for _key, _values in self._episode_max_values.items():",
        f"{indent}        if isinstance(_values, torch.Tensor):",
        f"{indent}            self.extras['log'][_key] = _values[env_ids].mean().item()",
        f"{indent}            self._episode_max_values[_key][env_ids] = 0.0",
    ]
    return mean_block + final_block + max_block


def _is_super_reset_idx_call(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "_reset_idx"
    )


def _detect_env_class(tree: ast.Module) -> str | None:
    for node in reversed(tree.body):
        if (
            isinstance(node, ast.ClassDef)
            and node.name.endswith("Env")
            and not node.name.endswith("EnvCfg")
        ):
            return node.name
    return None


def _detect_cfg_class(tree: ast.Module) -> str | None:
    for node in reversed(tree.body):
        if isinstance(node, ast.ClassDef) and node.name.endswith("EnvCfg"):
            return node.name
    return None


def _find_method(tree: ast.Module, method_name: str) -> ast.FunctionDef | None:
    env_class_name = _detect_env_class(tree)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == env_class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    return None


def _find_reset_idx(tree: ast.Module) -> ast.FunctionDef | None:
    return _find_method(tree, "_reset_idx")


def _method_references_generated_metrics(func_node: ast.FunctionDef) -> bool:
    tracked = ("_episode_sums", "_episode_final_values", "_episode_max_values")
    for node in ast.walk(func_node):
        if isinstance(node, ast.Attribute) and node.attr in tracked:
            return True
        if isinstance(node, ast.Name) and node.id in tracked:
            return True
    return False
