"""Sample env config parameters from *_bounds fields and write a candidate env."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path


def create_candidate_env(base_env_path: Path, run_dir: Path, seed: int | None = None) -> Path:
    """Sample all *_bounds fields, write <run_dir>/candidate_env/candidate_env.py.

    The *_bounds fields remain in the output unchanged.  Only the corresponding
    active fields (same name without the _bounds suffix) have their values
    replaced with freshly sampled ones.  Returns the path to candidate_env.py.
    """
    if seed is not None:
        random.seed(seed)

    source = base_env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    replacements = _collect_replacements(tree)

    # Apply in reverse line order so earlier positions stay valid.
    for value_node, new_val in sorted(replacements, key=lambda x: x[0].lineno, reverse=True):
        source = _replace_node(source, value_node, new_val)

    candidate_dir = run_dir / "candidate_env"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = candidate_dir / "candidate_env.py"
    candidate_path.write_text(source, encoding="utf-8")
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
# Internals
# ---------------------------------------------------------------------------

def _collect_replacements(tree: ast.Module) -> list[tuple[ast.expr, float | int]]:
    """Return (value_node, sampled_value) for each active field that has a *_bounds twin."""
    cfg_class = _find_cfg_class(tree)
    if cfg_class is None:
        return []

    fields: dict[str, ast.AnnAssign] = {}
    for stmt in cfg_class.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields[stmt.target.id] = stmt

    results = []
    for name, node in fields.items():
        if not name.endswith("_bounds"):
            continue
        active_name = name[: -len("_bounds")]
        active_node = fields.get(active_name)
        if active_node is None or active_node.value is None:
            continue
        bounds = _extract_bounds(node.value)
        if bounds is None:
            continue
        results.append((active_node.value, _sample(bounds)))

    return results


def _extract_bounds(node: ast.expr) -> tuple[float | int, float | int] | None:
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        lo, hi = node.elts
        if isinstance(lo, ast.Constant) and isinstance(hi, ast.Constant):
            return (lo.value, hi.value)
    return None


def _sample(bounds: tuple[float | int, float | int]) -> float | int:
    lo, hi = bounds
    if isinstance(lo, int) and isinstance(hi, int):
        return random.randint(lo, hi)
    return random.uniform(float(lo), float(hi))


def _replace_node(source: str, node: ast.expr, new_val: float | int) -> str:
    """Replace the source span of node with the string representation of new_val."""
    lines = source.splitlines(keepends=True)
    line = lines[node.lineno - 1]
    text = str(new_val) if isinstance(new_val, int) else repr(round(float(new_val), 6))
    lines[node.lineno - 1] = line[: node.col_offset] + text + line[node.end_col_offset :]
    return "".join(lines)


def _find_cfg_class(tree: ast.Module) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("EnvCfg"):
            return node
    return None


def _detect_env_class(tree: ast.Module) -> str | None:
    for node in reversed(tree.body):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Env") and not node.name.endswith("EnvCfg"):
            return node.name
    return None


def _detect_cfg_class(tree: ast.Module) -> str | None:
    for node in reversed(tree.body):
        if isinstance(node, ast.ClassDef) and node.name.endswith("EnvCfg"):
            return node.name
    return None
