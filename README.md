# Unitree G1 Locomotion and Goal Navigation

This project implements PPO-based reinforcement learning for Unitree G1 locomotion tasks in IsaacLab.

It contains two tasks:

- `Isaac-Velocity-Flat-G1-v0`: velocity tracking on flat terrain.
- `G1Goal-v0`: a custom goal navigation task where G1 moves toward a target position.

The project includes PPO training, rollout collection, trajectory buffer, GAE computation, policy/value networks, checkpoint saving, and policy playback.

## Project Structure

```text
.
├── algorithms/        # PPO training and playback algorithms
├── buffers/           # Trajectory buffer for on-policy rollouts
├── configs/           # Training and playback configuration files
├── envs/              # Custom G1 goal navigation environment and rewards
├── modules/           # Policy, value, and MLP network modules
├── utility/           # Environment creation, parameter loading, logger, PPO utils
├── main.py            # Main entry for training and playback
└── README.md
````

## Requirements

This project requires IsaacLab and Isaac Sim.

Please make sure IsaacLab is correctly installed and the official G1 tasks can run before using this project.

Main dependencies:

* Python
* PyTorch
* IsaacLab
* Isaac Sim
* NumPy
* PyYAML
* TensorBoard

## Training

Train the velocity tracking task:

```bash
python main.py -cfg configs/g1_velocity.yaml --headless
```

Train the goal navigation task:

```bash
python main.py -cfg configs/g1_goal.yaml --headless
```

## Playback

Play the velocity tracking policy:

```bash
python main.py -cfg configs/g1_velocity_play.yaml --livestream 2
```

Play the goal navigation policy:

```bash
python main.py -cfg configs/g1_goal_play.yaml --livestream 2
```

Before playback, set `load_model_dir` in the corresponding play config file to the checkpoint path.

Example:

```yaml
load_model_dir: ./outputs/G1Goal-v0_PPO/2026-05-03/17-37-01/model/_reward32.3_399
```
