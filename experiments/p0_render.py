import sys
import mujoco
import rl_x.environments.custom_mujoco.robot_locomotion.mjx_warp.environment as env_module
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_warp.viewer import MujocoViewer
from rl_x.runner.runner import Runner

PLANE_XML = "/home/keagan/projects/rlx/RL-X/rl_x/environments/custom_mujoco/robot_locomotion/robots/pink_p0/data/plane.xml"


class MeshViewer(MujocoViewer):
    def __init__(self, model, dt):
        self.full_model = mujoco.MjModel.from_xml_path(PLANE_XML)
        self.full_data = mujoco.MjData(self.full_model)
        super().__init__(self.full_model, dt)
        self.model = model


    def render(self, data):
        self.full_data.qpos[:] = data.qpos
        self.full_data.qvel[:] = data.qvel
        mujoco.mj_forward(self.full_model, self.full_data)
        stripped_model = self.model
        self.model = self.full_model
        super().render(self.full_data)
        self.model = stripped_model


env_module.MujocoViewer = MeshViewer

sys.argv = [sys.argv[0],
    "--algorithm.name=ppo.flax_full_jit",
    "--environment.name=custom_mujoco.robot_locomotion.mjx_warp",
    "--environment.train_robot=pink_p0",
    "--environment.terrain.type=plane",
    "--environment.nr_envs=1",
    "--environment.njmax=200",
    "--environment.naconmax_per_env=32",
    "--environment.render=True",
    "--runner.mode=test",
    "--runner.track_console=True",
    "--runner.track_tb=False",
    "--runner.track_wandb=False",
    "--runner.save_model=False",
    "--runner.load_model=/home/keagan/projects/rlx/RL-X/experiments/runs/placeholder/p0_locomotion/1790500036/models/latest.model",
]
Runner().run()
