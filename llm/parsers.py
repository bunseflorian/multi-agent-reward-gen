"""Parsers for Qwen model responses.

All parsing logic lives here: generic tag/code-block extraction plus the
role-specific helpers that turn raw text into structured pieces for the
planner, reviewer, critic, and coder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_THINK_RE = re.compile(r".*?</(?:think|thinking)>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</(?:think|thinking)>", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<code>.*?)```", re.DOTALL)

_PLANNER_FINAL_ANSWER_ERROR = "Planner output format error Increase output size or try again."


# ---------------------------------------------------------------------------
# Generic tag extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedRewardCode:
    """Parsed reward-code response."""

    raw_response: str
    code: str
    thinking: str | None = None


def extract_final_answer(raw_response: str) -> str:
    """Return text-agent output with thinking blocks stripped."""

    return _THINK_RE.sub("", raw_response).strip()


def require_final_answer_terminator(raw_response: str, agent_name: str) -> None:
    """No-op compatibility shim for text-agent output validation."""
    return None


def require_code_block_terminator(raw_response: str, agent_name: str) -> None:
    """Raise when a code-agent response does not end with a closed fenced code block."""

    stripped = _THINK_RE.sub("", raw_response).strip()
    if not stripped.endswith("```"):
        raise ValueError(_format_error(agent_name))
    matches = list(_CODE_BLOCK_RE.finditer(stripped))
    if not matches or matches[-1].end() != len(stripped):
        raise ValueError(_format_error(agent_name))


def parse_reward_code(raw_response: str) -> ParsedRewardCode:
    """Extract the last fenced Python reward-code block from a model response."""

    thinking_parts = [_extract_thinking_content(match.group(0)) for match in _THINK_RE.finditer(raw_response)]
    thinking = "\n\n".join(part for part in thinking_parts if part) or None

    without_thinking = _THINK_RE.sub("", raw_response).strip()
    python_blocks = [
        match.group("code").strip()
        for match in _CODE_BLOCK_RE.finditer(without_thinking)
        if match.group("lang").lower() in {"python", "py"}
    ]
    if not python_blocks:
        raise ValueError("Coder response did not contain a fenced Python code block.")

    return ParsedRewardCode(raw_response=raw_response, code=python_blocks[-1], thinking=thinking)


def extract_logged_output(raw_response: str) -> str:
    """Return the preferred output content for ``conversation.md``.

    Preference order:
    1. last fenced Python code block
    2. whole response with thinking blocks stripped
    """

    stripped = _THINK_RE.sub("", raw_response).strip()
    python_blocks = [
        match.group("code").strip()
        for match in _CODE_BLOCK_RE.finditer(stripped)
        if match.group("lang").lower() in {"python", "py"}
    ]
    if python_blocks:
        return python_blocks[-1]

    return stripped


def _extract_thinking_content(block: str) -> str:
    block = _THINK_CLOSE_RE.sub("", block, count=1)
    return block.strip()


def _format_error(agent_name: str) -> str:
    if agent_name.lower() == "planner":
        return _PLANNER_FINAL_ANSWER_ERROR
    return f"{agent_name} output format error. Increase output size or try again."


# ---------------------------------------------------------------------------
# Role-specific parsers (previously embedded in agents.py)
# ---------------------------------------------------------------------------

def extract_plan_bullets(text: str) -> list[str]:
    """Return Markdown bullet lines (without markers) from planner output."""

    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            bullets.append(stripped[2:].strip())
    return bullets


def extract_plan_tensor_contract(plan_text: str) -> list[str]:
    """Return tensor contract entries from the planner output.

    The Planner emits a compact ``Signal contract``. The first tensor on each
    line is the planned signal; additional ``self.*`` references on the same
    line are source tensors from ``derived from:``.
    """

    return _extract_plan_tensor_section(plan_text, "signal contract")


def _extract_plan_tensor_section(
    plan_text: str,
    section_name: str,
) -> list[str]:
    """Return self.* entries from a planner tensor section."""

    entries: list[str] = []
    in_section = False
    for line in plan_text.splitlines():
        stripped = line.strip()
        header = stripped.lstrip("#").strip().rstrip(":").lower()
        if header.startswith(section_name):
            in_section = True
            continue
        if in_section and stripped.startswith("#"):
            break
        if in_section and stripped.endswith(":") and not stripped.startswith(("- ", "* ")):
            break
        if not in_section or not stripped.startswith(("- ", "* ")):
            continue
        match = re.search(
            r"(self\.[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)(?:\s*(\[shape:[^\]]+\]))?",
            stripped,
        )
        if not match:
            continue
        entry = match.group(1)
        if match.group(2):
            entry = f"{entry} {match.group(2)}"
        entries.append(entry)
        for ref in re.finditer(r"self\.[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*", stripped):
            path = ref.group(0)
            if path != match.group(1) and path not in entries:
                entries.append(path)
    return entries


def review_passed(final_answer_text: str) -> bool:
    """Return ``True`` when the reviewer explicitly marked the review as passing."""

    for line in final_answer_text.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith("review:"):
            return "pass" in normalized and "fail" not in normalized
    return False


def extract_review_feedback(final_answer_text: str) -> str:
    """Return concise feedback text from the reviewer's final answer."""

    lines = final_answer_text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("feedback:"):
            first = line.split(":", 1)[1].strip()
            rest = [item.strip() for item in lines[index + 1:] if item.strip()]
            return "\n".join([first, *rest]).strip()
    return final_answer_text.strip()


def critic_requests_revision(feedback: str, ppo_passed: bool) -> bool:
    """Return ``True`` when the critic wants a reward-code revision.

    Falls back to requesting a revision whenever PPO itself did not pass and
    the critic did not emit an explicit ``DECISION:`` line.
    """

    for line in feedback.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith("decision:"):
            decision = normalized.split(":", 1)[1].strip()
            if decision.startswith("revise"):
                return True
            if decision.startswith("accept"):
                return False
    return not ppo_passed
