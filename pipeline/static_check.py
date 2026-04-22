"""Static + mock-runtime validation for generated reward code.

Runs in three passes:

1. ``ast.parse`` (syntax sanity).
2. AST walks for forbidden imports, forbidden calls, and shape anti-patterns.
3. Mock execution against a minimal ``torch`` namespace to catch runtime
   shape/dtype mistakes in reward and metric tensors.
"""

from __future__ import annotations

import ast
import builtins
import re
from types import SimpleNamespace
from typing import Any

from ..core import TestReport


# ---------------------------------------------------------------------------
# Constants: what the reward body may and may not touch
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "pathlib", "shutil",
    "requests", "urllib", "typing",
}
_FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "globals", "input", "locals", "open"}
_FORBIDDEN_ATTR_ROOTS = {"os", "sys", "subprocess", "socket", "requests", "urllib", "pathlib", "shutil"}

_BASE_SELF_ATTRS = {
    "num_envs", "device", "cfg",
    "episode_length_buf", "step_dt", "max_episode_length",
    "reset_terminated", "reset_time_outs", "extras",
    "_episode_sums", "_episode_final_values", "_episode_max_values",
}
_BASE_DICT_METHODS = {"clear", "get", "items", "keys", "setdefault", "update", "values"}

_TENSOR_METHOD_SUFFIXES = (".float", ".double", ".long", ".int", ".bool", ".half", ".detach", ".clone")


ISAACLAB_TENSOR_SPECS: dict[str, tuple[int, ...]] = {
    "_robot.data.root_pos_w": (3,),
    "_robot.data.root_lin_vel_b": (3,),
    "_robot.data.root_ang_vel_b": (3,),
    "_robot.data.projected_gravity_b": (3,),
    "_robot.data.root_quat_w": (4,),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def static_check(
    code: str,
    allowed_tensors: list[str] | None = None,
    num_envs: int = 8,
) -> TestReport:
    """Validate reward code with an AST pass and a mock torch execution."""

    errors: list[str] = []
    warnings: list[str] = []

    tree = _parse(code, errors)
    if tree is not None:
        _validate_ast(tree, errors, code)
        _check_reward_assignment(tree, errors)
        if allowed_tensors is not None:
            _validate_self_attrs(tree, allowed_tensors, errors)
    if errors:
        return TestReport(False, errors=errors, warnings=warnings, metadata={"stage": "ast"})

    try:
        namespace = _mock_namespace(num_envs, allowed_tensors)
        exec(compile(code, "<generated_reward>", "exec"), _safe_globals(), namespace)
    except ImportError as exc:
        return TestReport(False, errors=[f"torch is required for mock tensor tests: {exc}"])
    except Exception as exc:
        return TestReport(
            False,
            errors=[f"Mock execution failed: {type(exc).__name__}: {exc}"],
            warnings=warnings,
            metadata={"stage": "exec"},
        )

    errors.extend(_reward_errors(namespace.get("reward"), num_envs))
    errors.extend(_metric_store_errors(namespace.get("self"), num_envs))
    return TestReport(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        metadata={"stage": "exec", "num_envs": num_envs},
    )


static_check.__test__ = False


# ---------------------------------------------------------------------------
# AST-level checks
# ---------------------------------------------------------------------------

def _parse(code: str, errors: list[str]) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        errors.append(_format_syntax_error(exc, code))
        return None


def _format_syntax_error(exc: SyntaxError, code: str) -> str:
    line = (exc.text or "").strip()
    if not line and exc.lineno is not None:
        lines = code.splitlines()
        if 1 <= exc.lineno <= len(lines):
            line = lines[exc.lineno - 1].strip()
    if line:
        return f"Syntax error: {exc.msg} near code: {line!r}"
    return f"Syntax error: {exc.msg}"


def _validate_ast(tree: ast.Module, errors: list[str], code: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_imports(node, errors)
        elif isinstance(node, ast.Call):
            _check_call(node, errors, code)
        elif isinstance(node, (ast.If, ast.IfExp)) and _test_uses_batched_self_tensor(node.test):
            snippet = ast.get_source_segment(code, node.test) or ast.dump(node.test)
            errors.append(
                "Python conditionals cannot branch on batched tensors. Use torch.where, torch.clamp, "
                f"or tensor masks instead near code: {snippet!r}"
            )
        elif isinstance(node, ast.Subscript) and _looks_like_batched_per_env_indexing(node):
            snippet = ast.get_source_segment(code, node) or ast.dump(node)
            errors.append(
                "Batched per-environment indexing must use env_ids = torch.arange(self.num_envs, "
                f"device=self.device) and tensor[env_ids, idx], not tensor[:, idx, :] near code: {snippet!r}"
            )
        elif isinstance(node, ast.Attribute):
            root = _root_name(node)
            if root in _FORBIDDEN_ATTR_ROOTS:
                errors.append(f"Forbidden attribute access: {root}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "num_envs":
            errors.append("Undefined variable: bare 'num_envs' is not available in _get_rewards; use self.num_envs.")


def _check_imports(node: ast.Import | ast.ImportFrom, errors: list[str]) -> None:
    imports = sorted({alias.name.split(".", 1)[0] for alias in node.names})
    forbidden = sorted(set(imports) & _FORBIDDEN_IMPORTS)
    if forbidden:
        errors.append(f"Forbidden import: {', '.join(forbidden)}. torch is already available in reward code.")
    elif imports:
        errors.append("Imports are not allowed in generated reward code; use only available runtime symbols.")


def _check_call(node: ast.Call, errors: list[str], code: str) -> None:
    name = _dotted_name(node.func)
    if name in _FORBIDDEN_CALLS or name.split(".", 1)[0] in _FORBIDDEN_ATTR_ROOTS:
        errors.append(f"Forbidden call: {name}")
    if name == "torch.dot":
        errors.append(
            "Forbidden call: torch.dot. Reward code uses batched tensors; "
            "use torch.sum(a * b, dim=-1) for per-environment dot products."
        )
    if _has_keepdim_true(node):
        snippet = ast.get_source_segment(code, node) or name
        errors.append(
            "keepdim=True is not allowed in generated reward code because reward components "
            f"and metrics must stay shape [num_envs]. Use .unsqueeze(-1) only for explicit "
            f"vector normalization near code: {snippet!r}"
        )
    if _is_cfg_scalar_method_call(node):
        snippet = ast.get_source_segment(code, node) or name
        errors.append(
            "Config scalar method calls are not allowed; config fields are Python scalars, "
            f"not tensors. Use the scalar value directly near code: {snippet!r}"
        )


def _check_reward_assignment(tree: ast.Module, errors: list[str]) -> None:
    """Fail early if the code never assigns to a variable literally named 'reward'."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "reward":
                    return
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "reward":
                return
    errors.append(
        "Generated code does not assign to variable 'reward'. "
        "The final reward must be assigned as: reward = <expression>"
    )


def _validate_self_attrs(tree: ast.Module, allowed_tensors: list[str], errors: list[str]) -> None:
    allowed = {_normalize_tensor_entry(entry) for entry in allowed_tensors}
    allowed.discard("")

    for path in sorted(_self_leaf_paths(tree)):
        if _is_base_self_path(path):
            continue
        if any(path == item or path.startswith(item + ".") for item in allowed):
            continue
        errors.append(f"Illegal self attribute: 'self.{path}' is not listed in Planner usable tensors.")


# ---------------------------------------------------------------------------
# Mock-runtime execution
# ---------------------------------------------------------------------------

def _safe_globals() -> dict[str, Any]:
    import torch
    return {"__builtins__": builtins, "torch": torch}


def _mock_namespace(num_envs: int, allowed_tensors: list[str] | None) -> dict[str, Any]:
    import torch

    specs = dict(ISAACLAB_TENSOR_SPECS)
    specs.update(_shape_specs_from_entries(allowed_tensors))

    env_self = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        cfg=SimpleNamespace(),
        extras={},
        _episode_sums={},
        _episode_final_values={},
        _episode_max_values={},
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        step_dt=0.02,
        max_episode_length=500,
        reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
        reset_time_outs=torch.zeros(num_envs, dtype=torch.bool),
    )

    for entry in allowed_tensors or []:
        path = _normalize_tensor_entry(entry)
        if not path or path in _BASE_SELF_ATTRS:
            continue
        shape = specs.get(path)
        value = _mock_value_for_path(path, shape, num_envs)
        _set_nested(env_self, path, value)

    return {"torch": torch, "self": env_self}


def _mock_value_for_path(path: str, shape: tuple[int, ...] | None, num_envs: int) -> Any:
    import torch

    if path.startswith("cfg."):
        return 1
    tensor_shape = (num_envs,) if shape is None else (num_envs, *shape)
    leaf = path.rsplit(".", 1)[-1].lower()
    if any(token in leaf for token in ("success", "done", "terminated", "timeout", "flag", "mask")):
        return torch.zeros(tensor_shape, dtype=torch.bool)
    if leaf.endswith(("idx", "index")):
        return torch.zeros(tensor_shape, dtype=torch.long)
    return torch.zeros(tensor_shape)


def _reward_errors(reward: Any, num_envs: int) -> list[str]:
    import torch

    if reward is None:
        return ["Generated code did not define variable 'reward'."]
    if not isinstance(reward, torch.Tensor):
        return ["Variable 'reward' is not a torch.Tensor."]
    if tuple(reward.shape) != (num_envs,):
        return [f"Variable 'reward' has shape {tuple(reward.shape)}, expected ({num_envs},)."]
    if not torch.isfinite(reward).all().item():
        return ["Variable 'reward' contains NaN or Inf values."]
    return []


def _metric_store_errors(env_self: Any, num_envs: int) -> list[str]:
    """Validate generated metric stores match reset-time logging requirements."""

    import torch

    errors: list[str] = []
    stores = {
        "_episode_sums": getattr(env_self, "_episode_sums", {}),
        "_episode_final_values": getattr(env_self, "_episode_final_values", {}),
        "_episode_max_values": getattr(env_self, "_episode_max_values", {}),
    }
    for store_name, store in stores.items():
        if not isinstance(store, dict):
            continue
        for key, value in store.items():
            if not isinstance(value, torch.Tensor):
                continue
            if tuple(value.shape) != (num_envs,):
                errors.append(
                    f"Metric '{key}' in {store_name} has shape {tuple(value.shape)}, "
                    f"expected ({num_envs},). Metrics must remain per-environment tensors."
                )
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                errors.append(
                    f"Metric '{key}' in {store_name} has dtype {value.dtype}; "
                    "metrics must be floating point because reset logging calls mean()."
                )
    return errors


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _has_keepdim_true(node: ast.Call) -> bool:
    return any(
        keyword.arg == "keepdim"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in node.keywords
    )


def _is_cfg_scalar_method_call(node: ast.Call) -> bool:
    """Return True for calls like ``self.cfg.some_value.float()``."""

    if not isinstance(node.func, ast.Attribute):
        return False
    receiver = node.func.value
    if not isinstance(receiver, ast.Attribute):
        return False
    return _dotted_name(receiver).startswith("self.cfg.")


def _test_uses_batched_self_tensor(test: ast.AST) -> bool:
    for child in ast.walk(test):
        if not isinstance(child, ast.Attribute):
            continue
        name = _dotted_name(child)
        if not name.startswith("self."):
            continue
        path = name[len("self.") :]
        if path.startswith("cfg.") or path in {"num_envs", "device"}:
            continue
        return True
    return False


def _looks_like_batched_per_env_indexing(node: ast.Subscript) -> bool:
    """Catch ``tensor[:, idx, :]`` style indexing with a dynamic per-env index."""

    if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) < 2:
        return False
    first, second = node.slice.elts[0], node.slice.elts[1]
    if not isinstance(first, ast.Slice):
        return False
    if isinstance(second, (ast.Constant, ast.Slice)):
        return False
    return True


def _self_leaf_paths(tree: ast.Module) -> set[str]:
    paths = {
        _dotted_name(node)[len("self.") :]
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and _dotted_name(node).startswith("self.")
    }
    return {path for path in paths if not any(other.startswith(path + ".") for other in paths)}


def _is_base_self_path(path: str) -> bool:
    if path in _BASE_SELF_ATTRS:
        return True
    root, _, method = path.partition(".")
    if root in {"extras", "_episode_sums", "_episode_final_values", "_episode_max_values"}:
        return method in _BASE_DICT_METHODS
    return False


def _normalize_tensor_entry(entry: str) -> str:
    token = entry.strip().lstrip("-*").strip().split(maxsplit=1)[0].rstrip(",;").removesuffix("()")
    token = token.removeprefix("self.")
    changed = True
    while changed:
        changed = False
        for suffix in _TENSOR_METHOD_SUFFIXES:
            for candidate in (suffix, suffix + "()"):
                if token.endswith(candidate):
                    token = token[: -len(candidate)]
                    changed = True
                    break
            if changed:
                break
    return token


def _shape_specs_from_entries(entries: list[str] | None) -> dict[str, tuple[int, ...]]:
    specs: dict[str, tuple[int, ...]] = {}
    for entry in entries or []:
        path = _normalize_tensor_entry(entry)
        shape = _shape_from_entry(entry)
        if path and shape is not None:
            specs[path] = shape
    return specs


def _shape_from_entry(entry: str) -> tuple[int, ...] | None:
    match = re.search(r"\[shape:\s*\(([^)]*)\)\]", entry)
    if not match:
        return None
    dims = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if not dims or dims[0] != "N":
        return None
    if not all(dim.isdigit() for dim in dims[1:]):
        return None
    return tuple(int(dim) for dim in dims[1:])


def _set_nested(root: SimpleNamespace, path: str, value: Any) -> None:
    current: Any = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = getattr(current, part, None)
        if not isinstance(child, SimpleNamespace):
            child = SimpleNamespace()
            setattr(current, part, child)
        current = child
    setattr(current, parts[-1], value)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Attribute):
        return type(node).__name__
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _root_name(node: ast.Attribute) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""
