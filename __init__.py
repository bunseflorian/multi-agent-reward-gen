# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import gymnasium as gym
    from . import agents
except (ImportError, ModuleNotFoundError):
    gym = None
    agents = None

if gym is not None:
    _ROOT = Path(__file__).resolve().parent
    _PPO_CANDIDATE_DIR = _ROOT / "runs" / "ppo_candidate"
    _MANIFEST = _PPO_CANDIDATE_DIR / "candidate_env" / "manifest.json"

    if _MANIFEST.is_file():
        _m = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        # The env file is always written as candidate_env/candidate_env.py
        _entry = "candidate_env.candidate_env"
        _candidate_parent = str(_PPO_CANDIDATE_DIR)
        if _candidate_parent not in sys.path:
            sys.path.insert(0, _candidate_parent)
        gym.register(
            id="qwen_agent_task",
            entry_point=f"{_entry}:{_m['env_class']}",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_entry}:{_m['env_cfg_class']}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
                "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            },
        )
