from modules import *
from algorithms import *


def alg_init(parameter, env=None):
    name = parameter.alg_name
    alg = None
    if name == "ppo":
        alg = PPO(parameter, env)
    elif name == "ppo_play":
        alg = PPOPlay(parameter, env)
    else:
        raise ValueError(f"Unsupported algorithm: {name}")
    return alg
