# Figure-8 (lemniscate) gate navigation environment.


from __future__ import annotations

import math

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

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterFigure8GateEnvWindow(BaseEnvWindow):
    """Window manager for the figure-8 gate quadcopter environment."""

    def __init__(self, env: "QuadcopterFigure8GateEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterFigure8GateEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 16.0
    decimation = 2
    action_space = 4
    observation_space = 14
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterFigure8GateEnvWindow

    # figure-8 (lemniscate) gate task
    num_waypoints = 8            # gates sampled along one full figure-8 loop
    gate_lateral_radius = 0.30   # horizontal slack, perpendicular to tangent [m]
    gate_height_tolerance = 0.30 # vertical slack around gate altitude [m] (now independent)
    gate_pass_min_vel = 0.2      # min forward speed along tangent to count a pass [m/s]
    figure8_scale = 1.5          # size ("radius") of the figure-8, in meters
    figure8_height = 1.0         # fixed flight height for the pattern
    tangent_dt = 0.01            # small step used to estimate gate facing direction

    # reward scales
    dist_reward_scale = 2.0
    gate_pass_reward_scale = 5.0
    success_reward_scale = 100.0
    alignment_reward_scale = 2.0   # rewards flying toward gate in correct direction, weighted by proximity
    lin_vel_reward_scale = -0.05   # penalizes translational energy
    ang_vel_reward_scale = -0.01

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


class QuadcopterFigure8GateEnv(DirectRLEnv):
    cfg: QuadcopterFigure8GateEnvCfg

    def __init__(self, cfg: QuadcopterFigure8GateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Gate positions + facing directions (world frame). Normals are horizontal
        # (z-component always 0) since the figure-8 tangent is computed in the XY plane.
        self._waypoints_w = torch.zeros(self.num_envs, self.cfg.num_waypoints, 3, device=self.device)
        self._waypoints_normal_w = torch.zeros(self.num_envs, self.cfg.num_waypoints, 3, device=self.device)
        self._active_waypoint_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._active_waypoint_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waypoints_completed = torch.zeros(self.num_envs, device=self.device)
        self._waypoint_progress = torch.zeros(self.num_envs, device=self.device)
        self._distance_to_waypoint = torch.zeros(self.num_envs, device=self.device)
        self._waypoint_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Signed distance (along tangent, horizontal-only) to the active gate's plane,
        # from the previous step. Used to detect the moment the drone crosses through.
        self._prev_signed_dist = torch.zeros(self.num_envs, device=self.device)

        # Reward-input placeholders, filled in by _get_rewards() each step and read
        # by _compute_generated_reward() -- kept on self so the reward method takes
        # only self, matching the qwen_agentv2 pipeline's zero-argument calling
        # convention (see env_context.py / env_patch.py: both look up the method by
        # the exact name _compute_generated_reward, taking only self).
        self._rel_to_gate = torch.zeros(self.num_envs, 3, device=self.device)
        self._gate_normal_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._drone_vel_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Robot physical parameters.
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        # NOTE: verify this matches the Articulation API in your installed IsaacLab
        # version (grep isaaclab/assets/articulation/articulation.py if unsure).
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
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        drone_pos = self._robot.data.root_pos_w
        drone_vel = self._robot.data.root_lin_vel_w

        rel = drone_pos - self._active_waypoint_w              # (N, 3)
        rel_xy = rel[:, :2]
        normal_xy = self._active_waypoint_normal_w[:, :2]       # unit, horizontal

        # Signed distance along tangent (horizontal only) — used for crossing detection.
        signed_dist = torch.sum(rel_xy * normal_xy, dim=-1)     # (N,)

        # Horizontal lateral offset, perpendicular to tangent — independent of height.
        lateral_xy = rel_xy - signed_dist.unsqueeze(-1) * normal_xy
        lateral_dist = torch.norm(lateral_xy, dim=-1)

        # Vertical offset, checked separately from lateral (this is the fix).
        height_err = torch.abs(rel[:, 2])

        # Forward speed through the gate plane, along the tangent direction.
        vel_axial = torch.sum(drone_vel[:, :2] * normal_xy, dim=-1)

        crossed_plane = torch.logical_and(self._prev_signed_dist < 0.0, signed_dist >= 0.0)
        within_lateral = lateral_dist <= self.cfg.gate_lateral_radius
        within_height = height_err <= self.cfg.gate_height_tolerance
        fast_enough = vel_axial >= self.cfg.gate_pass_min_vel

        self._waypoint_reached = (
            crossed_plane & within_lateral & within_height & fast_enough & ~self._task_success
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

        # Update active gate (may have advanced this step).
        env_idx = torch.arange(self.num_envs, device=self.device)
        self._active_waypoint_w = self._waypoints_w[env_idx, self._current_waypoint_idx]
        self._active_waypoint_normal_w = self._waypoints_normal_w[env_idx, self._current_waypoint_idx]

        # Re-baseline the signed distance against the (possibly new) active gate.
        rel_new = drone_pos - self._active_waypoint_w
        normal_new_xy = self._active_waypoint_normal_w[:, :2]
        self._prev_signed_dist = torch.sum(rel_new[:, :2] * normal_new_xy, dim=-1)
        self._distance_to_waypoint = torch.norm(rel_new, dim=-1)

        # Store on self, not passed as arguments -- pipeline-compatible calling
        # convention: env_context.py / env_patch.py expect _compute_generated_reward
        # to take only self, reading everything it needs off self.something.
        self._rel_to_gate = rel_new
        self._gate_normal_xy = normal_new_xy
        self._drone_vel_w = drone_vel

        return self._compute_generated_reward()

    def _compute_generated_reward(self) -> torch.Tensor:
        # QWEN-GENERATED REWARD FUNCTION

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        task_success = self._task_success
        crash = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 2.5,
        )
        terminated = torch.logical_or(task_success, crash)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._current_waypoint_idx[env_ids] = 0
        self._waypoints_completed[env_ids] = 0.0
        self._waypoint_progress[env_ids] = 0.0
        self._distance_to_waypoint[env_ids] = 0.0
        self._waypoint_reached[env_ids] = False
        self._task_success[env_ids] = False
        self._sample_waypoints(env_ids)
        self._active_waypoint_w[env_ids] = self._waypoints_w[env_ids, 0]
        self._active_waypoint_normal_w[env_ids] = self._waypoints_normal_w[env_ids, 0]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Baseline the signed-distance tracker against gate 0, using the
        # just-reset position, so the very first crossing check is correct.
        reset_pos = default_root_state[:, :3]
        rel0_xy = (reset_pos - self._waypoints_w[env_ids, 0])[:, :2]
        normal0_xy = self._waypoints_normal_w[env_ids, 0, :2]
        self._prev_signed_dist[env_ids] = torch.sum(rel0_xy * normal0_xy, dim=-1)

    # -------------------------------------------------------------------
    # FIGURE-8 GATE SAMPLING: position + facing direction per gate
    # -------------------------------------------------------------------
    def _sample_waypoints(self, env_ids: torch.Tensor):
        """Generate gates along a figure-8 (lemniscate of Bernoulli).

        Each gate gets:
        - a position on the curve
        - a facing direction (the curve's tangent at that point), used to
          define the gate's "plane" for the crossing check in _get_rewards.

        A small random phase offset per environment means the drone doesn't
        always start at the exact same point on the curve.
        """
        num_resets = len(env_ids)
        num_wp = self.cfg.num_waypoints
        a = self.cfg.figure8_scale
        dt = self.cfg.tangent_dt

        base_t = torch.linspace(0.0, 2.0 * math.pi, steps=num_wp + 1, device=self.device)[:-1]
        phase = torch.empty(num_resets, 1, device=self.device).uniform_(0.0, 2.0 * math.pi)
        t = base_t.unsqueeze(0) + phase  # (num_resets, num_wp)

        def lemniscate_xy(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            sin_a = torch.sin(angle)
            cos_a = torch.cos(angle)
            denom = 1.0 + sin_a * sin_a
            return a * cos_a / denom, a * sin_a * cos_a / denom

        x, y = lemniscate_xy(t)
        x_ahead, y_ahead = lemniscate_xy(t + dt)

        z = torch.full_like(x, self.cfg.figure8_height)
        waypoints = torch.stack([x, y, z], dim=-1)  # (num_resets, num_wp, 3)
        waypoints[:, :, :2] += self._terrain.env_origins[env_ids, None, :2]
        self._waypoints_w[env_ids] = waypoints

        # Tangent (facing) direction: horizontal only, unit length.
        dx = x_ahead - x
        dy = y_ahead - y
        tangent_len = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
        normals = torch.stack([dx / tangent_len, dy / tangent_len, torch.zeros_like(dx)], dim=-1)
        self._waypoints_normal_w[env_ids] = normals

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