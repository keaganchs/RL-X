import os
import sys
from pathlib import Path

ISAAC_LAB_PYTHON = "/home/keagan/projects/isaac/IsaacLab/.venv/bin/python"

if Path(sys.prefix).resolve() != Path(ISAAC_LAB_PYTHON).parent.parent.resolve():
    os.execv(ISAAC_LAB_PYTHON, [ISAAC_LAB_PYTHON] + sys.argv)

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

if len(sys.argv) > 1:
    model_path = Path(sys.argv[1]).resolve()
else:
    candidates = sorted(Path(__file__).parent.glob("runs/*/p0_isaac_locomotion/*/models/best.model"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        sys.exit("No checkpoint found in runs/*/p0_isaac_locomotion/*/models/best.model. Pass a .model path as the first argument.")
    model_path = candidates[-1]
print(f"Rendering {model_path}")

from rl_x.runner.runner import Runner

sys.argv = [sys.argv[0],
    "--algorithm.name=ppo.pytorch",
    "--environment.name=custom_isaac_lab.p0_locomotion",
    "--environment.nr_envs=1",
    "--environment.render=True",
    "--runner.mode=test",
    "--runner.nr_test_episodes=1000000",
    "--runner.track_console=True",
    "--runner.track_tb=False",
    "--runner.track_wandb=False",
    "--runner.save_model=False",
    f"--runner.load_model={model_path}",
]
Runner().run()
