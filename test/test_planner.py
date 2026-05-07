"""Run the Planner against the tracking env file + a short user prompt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, call_llm, load_prompt, print_banner  # noqa: E402


ENV_FILE = ROOT / "environments" / "quadcopter_env_tracking.py"
USER_PROMPT = "Minimize distance to the target position."


def main() -> None:
    system_prompt = load_prompt("PLANNER_SYSTEM_PROMPT")
    user_prompt = "\n".join(
        [
            "Task ID: quadcopter_env_tracking",
            "",
            "User Prompt:",
            USER_PROMPT,
            "",
            f"Environment source path: {ENV_FILE}",
            "",
            "Relevant environment code:",
            ENV_FILE.read_text(),
            "",
        ]
    )
    output = call_llm(system_prompt, user_prompt, max_tokens=2048)
    print_banner("PLANNER OUTPUT")
    print(output)


if __name__ == "__main__":
    main()
