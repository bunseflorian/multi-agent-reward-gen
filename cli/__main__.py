"""``python -m qwen_agentv2`` entry point.

Also handles the optional Qwen-server readiness check previously owned by
``runtime.multi_agent_runner``. Pass ``--check-server`` to mimic that flow.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

from ..configs import get_path
from ..core import PPOConfig
from ..llm import QwenApiError
from ..pipeline import run_dir_name, save_run_state
from ..server import qwen_server_is_ready
from .display import print_state
from .run import run_task


LOGGER = logging.getLogger("qwen_agentv2")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging()

    if args.check_server:
        if not _server_precheck():
            return 1

    env_path = Path(args.env_path) if args.env_path else get_path("environment", "default_env")
    output_root = Path(args.output_dir) if args.output_dir else get_path("runtime", "runs_dir")

    tmp_run_dir = output_root / f"_running_{env_path.stem}"
    tmp_run_dir.mkdir(parents=True, exist_ok=True)
    effective_run_dir: Path = Path(args.ppo_run_dir) if args.ppo_run_dir else tmp_run_dir

    try:
        state = run_task(
            max_iterations=args.max_iterations,
            max_rl_iterations=args.max_rl_iterations,
            env_path=env_path,
            objective=args.objective,
            ppo_config=PPOConfig(
                run_rl_training=args.run_rl_training,
                num_envs=args.ppo_num_envs,
                max_iterations=args.ppo_max_iterations,
                timeout_seconds=args.ppo_timeout_seconds,
                headless=args.ppo_headless,
            ),
            run_dir=effective_run_dir,
        )
    except QwenApiError as exc:
        print(f"Multi-agent run failed because Qwen is unavailable: {exc}", file=sys.stderr)
        return 1

    if args.ppo_run_dir is None:
        final_run_dir = output_root / run_dir_name(state)
        tmp_run_dir.rename(final_run_dir)
        save_run_state(state, run_dir=final_run_dir)
    print_state(state)
    return 0 if state.accepted else 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local multi-agent reward-code generator.")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-rl-iterations", type=int, default=None)
    parser.add_argument("--env-path", type=Path, default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-rl-training", action="store_true", default=False)
    parser.add_argument("--ppo-num-envs", type=int, default=None)
    parser.add_argument("--ppo-max-iterations", type=int, default=None)
    parser.add_argument("--ppo-timeout-seconds", type=int, default=None)
    parser.add_argument("--ppo-headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ppo-run-dir", type=Path, default=None)
    parser.add_argument(
        "--check-server",
        action="store_true",
        default=False,
        help="Verify the Qwen server is reachable before running.",
    )
    return parser.parse_args(argv)


def _server_precheck() -> bool:
    missing = [name for name in ("torch",) if importlib.util.find_spec(name) is None]
    if missing:
        LOGGER.error("Missing required packages: %s", ", ".join(missing))
        return False
    base_url = os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    os.environ.setdefault("QWEN_BASE_URL", base_url)
    if not qwen_server_is_ready(base_url):
        LOGGER.error("Qwen server is not reachable at %s", base_url)
        LOGGER.error("Start it first: python -m qwen_agentv2.server.launcher")
        return False
    LOGGER.info("Qwen server is reachable at %s", base_url)
    return True


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
