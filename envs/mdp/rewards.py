from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time_positive_biped_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    distance_threshold: float = 0.2,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    pose_cmd = env.command_manager.get_command(command_name)
    reward = reward * (torch.norm(pose_cmd[:, :2], dim=1) > distance_threshold)
    return reward


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.3,
) -> torch.Tensor:
    pose_cmd = env.command_manager.get_command(command_name)
    distance = torch.norm(pose_cmd[:, :2], dim=1)
    return (distance < threshold).float()


def stand_still_at_goal(
    env,
    command_name: str,
    threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    pose_cmd = env.command_manager.get_command(command_name)
    distance = torch.norm(pose_cmd[:, :2], dim=1)
    near = distance < threshold
    lin_vel_xy = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    ang_vel_z = torch.abs(asset.data.root_ang_vel_b[:, 2])
    quiet = torch.exp(-(lin_vel_xy + 0.5 * ang_vel_z) / 0.25)
    return near.float() * quiet
