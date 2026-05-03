import gymnasium as gym
from .g1_goal import G1GoalCfg

gym.register(
    "G1Goal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_goal:G1GoalCfg",
    },
)
