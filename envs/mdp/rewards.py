# Copyright (c) 2022-2026, ...
# SPDX-License-Identifier: BSD-3-Clause

"""Custom rewards for G1 goal-reaching task (no velocity command dependency)."""

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
    """Biped feet-air-time reward gated by distance to a pose goal.

    与 :func:`feet_air_time_positive_biped` 完全等价，区别只在最后的 gating：
    原版通过速度命令的幅值 ``||cmd[:, :2]|| > 0.1`` 判断"是否需要行走"，
    本函数通过 ``pose_command`` 的位置分量幅值（即到目标的距离）
    ``||pose_cmd[:, :2]|| > distance_threshold`` 判断同样的事情。

    含义：离目标足够远时，奖励单脚支撑 + 抬脚（最长不超过 threshold）；
    临近目标（< distance_threshold）时不再奖励步态，让 fine-grained 的
    position tracking reward 主导，避免机器人在目标点附近还要"踏步"。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # 单脚支撑判定（人形步态的关键约束）
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # gating: 离目标足够远才激活
    pose_cmd = env.command_manager.get_command(command_name)
    reward = reward * (torch.norm(pose_cmd[:, :2], dim=1) > distance_threshold)
    return reward


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Dense bonus for being within ``threshold`` of the goal.

    给一个 0/1 的 bonus（不终止 episode）。比起 ``position_command_error_tanh``
    的连续奖励，这个可以让"完全到达"和"差不多到达"在 return 上有明显的台阶差，
    经验上能加快收敛。注意：weight 不要给太大，否则会和 fine-grained tracking 抢梯度。
    """
    pose_cmd = env.command_manager.get_command(command_name)
    distance = torch.norm(pose_cmd[:, :2], dim=1)
    return (distance < threshold).float()


def stand_still_at_goal(
    env,
    command_name: str,
    threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """到达目标且 base 速度小时给奖励。"""
    asset = env.scene[asset_cfg.name]
    pose_cmd = env.command_manager.get_command(command_name)
    distance = torch.norm(pose_cmd[:, :2], dim=1)
    near = distance < threshold
    # base 在世界系下的水平速度
    lin_vel_xy = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    ang_vel_z = torch.abs(asset.data.root_ang_vel_b[:, 2])
    quiet = torch.exp(-(lin_vel_xy + 0.5 * ang_vel_z) / 0.25)
    return near.float() * quiet
