import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import logging

logger = logging.getLogger("unrl.make_env")


def _resolve_device(parameter, device=None):
    if device is not None:
        return device
    if hasattr(parameter, "device"):
        return parameter.device
    if hasattr(parameter, "device_id"):
        return f"cuda:{parameter.device_id}"
    return "cuda:0"


def make_env(parameter, device=None):
    device = _resolve_device(parameter, device)
    env_cfg = parse_env_cfg(
        task_name=parameter.task,
        num_envs=parameter.num_envs,
        use_fabric=parameter.use_fabric,
        device=device,
    )

    env = gym.make(parameter.task, cfg=env_cfg)
    logger.info(f"Created env {parameter.task}")
    return env
