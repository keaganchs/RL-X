# Custom Isaac Lab Environments

Contains examples for custom Isaac Lab environments.

The Ant example uses the Ant robot and defines as the task to reach a far away target position.

The P0 Locomotion example trains the Pink Robotics P0 humanoid to track velocity commands on flat ground. It uses the same robot model as the [Robot Locomotion](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_locomotion) MuJoCo environments (```robots/pink_p0/data/p0_isaac_lab.urdf```), with the same foot contact boxes, PD gains and home pose.

| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Ant | Flat value | Continuous | Torch |
| P0 Locomotion | Flat value | Continuous | Torch |
