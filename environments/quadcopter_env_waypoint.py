# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterWaypointEnvWindow(BaseEnvWindow):
    """Window manager for the waypoint quadcopter environment."""

    def __init__(self, env: "QuadcopterWaypointEnv", window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterWaypointEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 16.0
    decimation = 2
    action_space = 4
    observation_space = 14
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterWaypointEnvWindow

    # randomizable task parameters
    num_waypoints: int = 5
    num_waypoints_bounds: tuple[int, int] = (3, 8)

    waypoint_xy_range: float = 2.0
    waypoint_xy_range_bounds: tuple[float, float] = (1.0, 3.0)

    waypoint_z_min: float = 0.5
    waypoint_z_min_bounds: tuple[float, float] = (0.3, 0.8)

    waypoint_z_max: float = 1.5
    waypoint_z_max_bounds: tuple[float, float] = (1.0, 2.5)

    waypoint_reached_threshold: float = 0.3
    waypoint_reached_threshold_bounds: tuple[float, float] = (0.15, 0.5)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01


class QuadcopterWaypointEnv(DirectRLEnv):
    cfg: QuadcopterWaypointEnvCfg

    def __init__(self, cfg: QuadcopterWaypointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions and applied forces (actions -> thrust/moment).
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Waypoint task state (world frame).
        self._waypoints_w = torch.zeros(self.num_envs, self.cfg.num_waypoints, 3, device=self.device)
        self._active_waypoint_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waypoints_completed = torch.zeros(self.num_envs, device=self.device)
        self._waypoint_progress = torch.zeros(self.num_envs, device=self.device)
        self._distance_to_waypoint = torch.zeros(self.num_envs, device=self.device)
        self._waypoint_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Robot physical parameters.
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Debug visualization (set to a valid handle inside set_debug_vis).
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        active_waypoint_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._active_waypoint_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                active_waypoint_b,
                self._waypoint_progress.unsqueeze(-1),
                self._task_success.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Update waypoint state before evaluating the generated reward.
        distance_to_previous_waypoint = torch.norm(self._robot.data.root_pos_w - self._active_waypoint_w, dim=-1)
        self._waypoint_reached = torch.logical_and(
            distance_to_previous_waypoint <= self.cfg.waypoint_reached_threshold, ~self._task_success
        )

        next_waypoint_idx = self._current_waypoint_idx + self._waypoint_reached.long()
        self._task_success = torch.logical_or(self._task_success, next_waypoint_idx >= self.cfg.num_waypoints)
        self._current_waypoint_idx = torch.clamp(next_waypoint_idx, max=self.cfg.num_waypoints - 1)
        self._waypoints_completed = torch.where(
            self._task_success,
            torch.full_like(self._waypoints_completed, float(self.cfg.num_waypoints)),
            self._current_waypoint_idx.float(),
        )
        self._waypoint_progress = self._waypoints_completed / float(self.cfg.num_waypoints)
        self._active_waypoint_w = self._waypoints_w[
            torch.arange(self.num_envs, device=self.device), self._current_waypoint_idx
        ]
        self._distance_to_waypoint = torch.norm(self._robot.data.root_pos_w - self._active_waypoint_w, dim=-1)

        reward = self._compute_generated_reward()
        return reward

    def _compute_generated_reward(self) -> torch.Tensor:
        # QWEN-GENERATED REWARD FUNCTION

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        task_success = self._task_success
        crash = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )

        terminated = torch.logical_or(task_success, crash)
        return terminated, time_out
    

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset per-env state + sample a new waypoint sequence.
        self._actions[env_ids] = 0.0
        self._current_waypoint_idx[env_ids] = 0
        self._waypoints_completed[env_ids] = 0.0
        self._waypoint_progress[env_ids] = 0.0
        self._distance_to_waypoint[env_ids] = 0.0
        self._waypoint_reached[env_ids] = False
        self._task_success[env_ids] = False
        self._sample_waypoints(env_ids)
        self._active_waypoint_w[env_ids] = self._waypoints_w[env_ids, 0]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _sample_waypoints(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        xy = torch.empty(num_resets, self.cfg.num_waypoints, 2, device=self.device).uniform_(
            -self.cfg.waypoint_xy_range, self.cfg.waypoint_xy_range
        )
        z = torch.empty(num_resets, self.cfg.num_waypoints, 1, device=self.device).uniform_(
            self.cfg.waypoint_z_min, self.cfg.waypoint_z_max
        )
        waypoints = torch.cat([xy, z], dim=-1)
        waypoints[:, :, :2] += self._terrain.env_origins[env_ids, None, :2]
        self._waypoints_w[env_ids] = waypoints

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._active_waypoint_w)
