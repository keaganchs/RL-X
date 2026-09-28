from pathlib import Path


def environment_creator(config):

    import gymnasium as gym
    import torch
    import warp as wp
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass, index_fill_


    urdf_path = Path(__file__).parent.parent.parent / "custom_mujoco" / "robot_locomotion" / "robots" / "pink_p0" / "data" / "p0_isaac_lab.urdf"

    P0_CFG = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path.as_posix(),
            fix_base=False,
            merge_fixed_joints=True,
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=[
                PhysxArticulationCfg(enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0),
                NewtonArticulationCfg(self_collision_enabled=False),
            ],
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=2.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.92),
            joint_pos={
                ".*_hip_pitch_joint": 0.2,
                ".*_knee_joint": 0.4,
                ".*_ankle_pitch_joint": 0.2,
                ".*_shoulder_roll_joint": -0.2,
                ".*_elbow_pitch_joint": 0.3,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=100.0,
                damping=2.0,
                effort_limit_sim={
                    "torso_.*|.*_hip_.*|.*_knee_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint": 120.0,
                    ".*_ankle_pitch_joint|.*_elbow_pitch_joint": 36.0,
                    ".*_ankle_roll_joint|.*_shoulder_yaw_joint": 14.0,
                    "head_.*|.*_elbow_roll_joint": 5.5,
                },
                armature={
                    "torso_.*|.*_hip_.*|.*_knee_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint": 0.02,
                    ".*_ankle_.*|.*_elbow_.*|.*_shoulder_yaw_joint|head_.*": 0.01,
                },
            ),
        },
    )


    @configclass
    class P0SceneCfg(InteractiveSceneCfg):
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
        robot = P0_CFG
        contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.005, track_air_time=True)
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))


    @configclass
    class P0EnvCfg(DirectRLEnvCfg):
        episode_length_s = 20.0
        decimation = 4
        action_scale = 1.0
        action_space = 26
        observation_space = 90
        state_space = 0

        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation, visualizer_cfgs=[KitVisualizerCfg(eye=(2.5, -2.5, 1.5), lookat=(0.0, 0.0, 0.7))] if config.environment.render else [])
        scene: P0SceneCfg = P0SceneCfg(num_envs=config.environment.nr_envs, env_spacing=3.0, replicate_physics=True)

        lin_vel_reward_scale = 1.0
        yaw_rate_reward_scale = 1.0
        z_vel_reward_scale = -2.0
        ang_vel_reward_scale = -0.05
        joint_torque_reward_scale = -1e-5
        joint_accel_reward_scale = -2.5e-7
        action_rate_reward_scale = -0.01
        feet_air_time_reward_scale = 0.25
        feet_air_time_threshold = 0.4
        feet_slide_reward_scale = -0.25
        flat_orientation_reward_scale = -1.0
        joint_deviation_reward_scale = -0.1
        joint_pos_limit_reward_scale = -1.0
        termination_reward_scale = -200.0

        termination_height = 0.6
        termination_gravity_z = -0.5


    class P0Locomotion(DirectRLEnv):
        cfg: P0EnvCfg


        def __init__(self, config):
            super().__init__(P0EnvCfg(), config.environment.rendering_mode)

            self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=float)
            self.single_observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.cfg.observation_space,), dtype=float)

            self.robot = self.scene["robot"]
            self.contact_sensor = self.scene["contact_sensor"]

            self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
            self.previous_actions = torch.zeros_like(self.actions)
            self.commands = torch.zeros(self.num_envs, 3, device=self.device)

            self.feet_ids, _ = self.contact_sensor.find_sensors(".*_ankle_roll_link_1")
            self.feet_body_ids, _ = self.robot.find_bodies(".*_ankle_roll_link_1")
            self.deviation_joint_ids, _ = self.robot.find_joints("torso_.*|head_.*|.*_shoulder_.*|.*_elbow_.*|.*_hip_roll_joint|.*_hip_yaw_joint|.*_ankle_roll_joint")

            self.reward_names = [
                "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2", "action_rate_l2",
                "feet_air_time", "feet_slide", "flat_orientation_l2", "joint_deviation_l1", "joint_pos_limits", "termination",
            ]
            self.episode_reward_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_names}
            self.episode_returns = torch.zeros(self.num_envs, device=self.device)
            self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
            self.last_episode_reward_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_names}
            self.last_episode_returns = torch.zeros(self.num_envs, device=self.device)
            self.last_episode_lengths = torch.zeros(self.num_envs, device=self.device)


        def _pre_physics_step(self, actions: torch.Tensor):
            self.actions = actions.clone()
            self.processed_actions = self.cfg.action_scale * self.actions + self.robot.data.default_joint_pos.torch


        def _apply_action(self):
            self.robot.set_joint_position_target_index(target=self.processed_actions)


        def _get_observations(self) -> dict:
            self.previous_actions = self.actions.clone()
            observation = torch.cat([
                self.robot.data.root_lin_vel_b.torch,
                self.robot.data.root_ang_vel_b.torch,
                self.robot.data.projected_gravity_b.torch,
                self.commands,
                self.robot.data.joint_pos.torch - self.robot.data.default_joint_pos.torch,
                self.robot.data.joint_vel.torch,
                self.actions,
            ], dim=-1)
            return {"policy": observation}


        def _get_rewards(self) -> torch.Tensor:
            root_lin_vel_b = self.robot.data.root_lin_vel_b.torch
            root_ang_vel_b = self.robot.data.root_ang_vel_b.torch
            joint_pos = self.robot.data.joint_pos.torch

            lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
            yaw_rate_error = torch.square(self.commands[:, 2] - root_ang_vel_b[:, 2])

            air_time = self.contact_sensor.data.current_air_time.torch[:, self.feet_ids]
            contact_time = self.contact_sensor.data.current_contact_time.torch[:, self.feet_ids]
            in_contact = contact_time > 0.0
            single_stance = torch.sum(in_contact.int(), dim=1) == 1
            in_mode_time = torch.where(in_contact, contact_time, air_time)
            feet_air_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0].clamp(max=self.cfg.feet_air_time_threshold)
            feet_air_time *= torch.linalg.norm(self.commands[:, :2], dim=1) > 0.1

            feet_contact = torch.max(torch.linalg.norm(self.contact_sensor.data.net_forces_w_history.torch[:, :, self.feet_ids], dim=-1), dim=1)[0] > 1.0
            feet_vel = torch.linalg.norm(self.robot.data.body_lin_vel_w.torch[:, self.feet_body_ids, :2], dim=-1)
            feet_slide = torch.sum(feet_vel * feet_contact, dim=1)

            soft_limits = self.robot.data.soft_joint_pos_limits.torch
            out_of_limits = -(joint_pos - soft_limits[..., 0]).clip(max=0.0) + (joint_pos - soft_limits[..., 1]).clip(min=0.0)

            rewards = {
                "track_lin_vel_xy_exp": torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale,
                "track_ang_vel_z_exp": torch.exp(-yaw_rate_error / 0.25) * self.cfg.yaw_rate_reward_scale,
                "lin_vel_z_l2": torch.square(root_lin_vel_b[:, 2]) * self.cfg.z_vel_reward_scale,
                "ang_vel_xy_l2": torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1) * self.cfg.ang_vel_reward_scale,
                "dof_torques_l2": torch.sum(torch.square(self.robot.data.applied_torque.torch), dim=1) * self.cfg.joint_torque_reward_scale,
                "dof_acc_l2": torch.sum(torch.square(self.robot.data.joint_acc.torch), dim=1) * self.cfg.joint_accel_reward_scale,
                "action_rate_l2": torch.sum(torch.square(self.actions - self.previous_actions), dim=1) * self.cfg.action_rate_reward_scale,
                "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale,
                "feet_slide": feet_slide * self.cfg.feet_slide_reward_scale,
                "flat_orientation_l2": torch.sum(torch.square(self.robot.data.projected_gravity_b.torch[:, :2]), dim=1) * self.cfg.flat_orientation_reward_scale,
                "joint_deviation_l1": torch.sum(torch.abs(joint_pos[:, self.deviation_joint_ids] - self.robot.data.default_joint_pos.torch[:, self.deviation_joint_ids]), dim=1) * self.cfg.joint_deviation_reward_scale,
                "joint_pos_limits": torch.sum(out_of_limits, dim=1) * self.cfg.joint_pos_limit_reward_scale,
                "termination": self.reset_terminated.float() * self.cfg.termination_reward_scale,
            }
            rewards = {name: value * self.step_dt for name, value in rewards.items()}
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            for name, value in rewards.items():
                self.episode_reward_sums[name] += value
            self.episode_returns += reward
            self.episode_lengths += 1

            return reward


        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            died = (self.robot.data.root_pos_w.torch[:, 2] < self.cfg.termination_height) | (self.robot.data.projected_gravity_b.torch[:, 2] > self.cfg.termination_gravity_z)
            return died, time_out


        def _reset_idx(self, env_ids: torch.Tensor | None):
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = wp.to_torch(self.robot._ALL_INDICES)
            self.robot.reset(env_ids)
            super()._reset_idx(env_ids)
            if len(env_ids) == self.num_envs:
                self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

            finished_env_ids = env_ids[self.episode_lengths[env_ids] > 0]
            self.last_episode_returns[finished_env_ids] = self.episode_returns[finished_env_ids]
            self.last_episode_lengths[finished_env_ids] = self.episode_lengths[finished_env_ids]
            for name in self.reward_names:
                self.last_episode_reward_sums[name][finished_env_ids] = self.episode_reward_sums[name][finished_env_ids]
                index_fill_(self.episode_reward_sums[name], env_ids, 0.0)
            index_fill_(self.episode_returns, env_ids, 0.0)
            index_fill_(self.episode_lengths, env_ids, 0.0)

            index_fill_(self.actions, env_ids, 0.0)
            index_fill_(self.previous_actions, env_ids, 0.0)
            commands = torch.zeros_like(self.commands[env_ids])
            commands[:, 0].uniform_(-1.0, 1.0)
            commands[:, 1].uniform_(-0.5, 0.5)
            commands[:, 2].uniform_(-1.0, 1.0)
            commands[torch.rand(len(env_ids), device=self.device) < 0.05] = 0.0
            self.commands[env_ids] = commands

            joint_pos = self.robot.data.default_joint_pos.torch[env_ids]
            joint_vel = self.robot.data.default_joint_vel.torch[env_ids]
            default_root_pose = self.robot.data.default_root_pose.torch[env_ids]
            default_root_vel = self.robot.data.default_root_vel.torch[env_ids]
            default_root_pose[:, :3] += self.scene.env_origins[env_ids]
            self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
            self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


        def get_info(self) -> dict:
            info = {"episode_return": self.last_episode_returns.cpu().numpy(), "episode_length": self.last_episode_lengths.cpu().numpy()}
            for name in self.reward_names:
                info[f"reward/{name}"] = self.last_episode_reward_sums[name].cpu().numpy()
            return info


        def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
            observation, reward, terminated, truncated, info = super().step(action)
            return observation["policy"], reward, terminated, truncated, self.get_info()


        def reset(self) -> tuple[torch.Tensor, dict]:
            observation, info = super().reset()
            return observation["policy"], self.get_info()


        def close(self):
            super().close()


        def get_logging_info_dict(self, info: dict) -> dict:
            return info


    return P0Locomotion(config)
