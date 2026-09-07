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
    """Replace ``_compute_generated_reward`` body with generated reward code.

    The Coder only computes ``reward``; per-component reward-term logging into
    ``self._episode_final_values`` is injected here so it stays consistent
    regardless of what the Coder emits.
    """

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

    logging_lines = [
        line
        for name in _reward_component_names(reward_code)
        for line in (
            f"{indent}self._episode_final_values.setdefault({_reward_term_key(name)!r}, torch.zeros_like({name}))",
            f"{indent}self._episode_final_values[{_reward_term_key(name)!r}] += {name}",
        )
    ]

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
        *logging_lines,
        "",
        f"{indent}return reward",
    ]
    return "\n".join(lines[: node.lineno] + body + lines[node.end_lineno:]) + "\n"


def _reward_component_names(reward_code: str) -> list[str]:
    """Return named locals referenced in the final ``reward = ...`` expression.

    These are the reward-term components the Coder composed in section 2 of
    the reward body; the patcher logs each one so the Coder never has to.
    """

    tree = ast.parse(reward_code)
    reward_assigns = [
        stmt for stmt in ast.walk(tree)
        if isinstance(stmt, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "reward" for target in stmt.targets)
    ]
    if not reward_assigns:
        return []
    value = reward_assigns[-1].value
    names = {
        node.id
        for node in ast.walk(value)
        if isinstance(node, ast.Name) and node.id not in {"reward", "torch"}
    }
    return sorted(names)


def _reward_term_key(component_name: str) -> str:
    return f"final_reward_term_{component_name}"


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


_GENERATED_METRIC_STORES = ("_episode_sums", "_episode_final_values", "_episode_max_values")


def patch_reset_idx_for_generated_metrics(source: str) -> str:
    """Inject generated-metric logging into ``_reset_idx``.

    The block is inserted just before the ``super()._reset_idx(...)`` call so
    dynamically-named metrics written by the Coder reach TensorBoard. Each of
    the three stores is handled independently: an env author may already flush
    one store by hand (e.g. their own ``_episode_sums`` bookkeeping) without
    knowing about the other two, so we must not skip all logging just because
    one store is already referenced -- that would silently drop, for example,
    every ``final_reward_term_*`` value the patcher accumulates into
    ``_episode_final_values``.
    """

    tree = ast.parse(source)
    reset_node = _find_reset_idx(tree)
    if reset_node is None or reset_node.end_lineno is None:
        return source

    missing_stores = [
        store for store in _GENERATED_METRIC_STORES
        if not _method_references_store(reset_node, store)
    ]
    if not missing_stores:
        return source

    indent = " " * (reset_node.col_offset + 4)
    block = _build_generated_metrics_block(indent, missing_stores)

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

def _build_generated_metrics_block(indent: str, stores: list[str]) -> list[str]:
    """Return the generic generated-metric logging block for the given stores only."""

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
    blocks_by_store = {
        "_episode_sums": mean_block,
        "_episode_final_values": final_block,
        "_episode_max_values": max_block,
    }
    return [line for store in stores for line in blocks_by_store[store]]


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


def _method_references_store(func_node: ast.FunctionDef, store: str) -> bool:
    for node in ast.walk(func_node):
        if isinstance(node, ast.Attribute) and node.attr == store:
            return True
        if isinstance(node, ast.Name) and node.id == store:
            return True
    return False
