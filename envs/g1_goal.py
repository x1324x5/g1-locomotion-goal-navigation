# Copyright (c) 2022-2026, ...
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg as VelocityCommandsCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1Rewards,
)

# 自定义 mdp（注意根据你实际放置位置调整 import 路径）
from .mdp import rewards as custom_rewards


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@configclass
class G1GoalCommandsCfg(VelocityCommandsCfg):
    """同时持有 base_velocity（占位，方便父类 __post_init__）和 pose_command。

    base_velocity 在 G1GoalCfg.__post_init__ 末尾被设为 None 关闭。
    """

    pose_command = nav_mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=nav_mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            heading=(-math.pi, math.pi),  # simple_heading=True 时此项被忽略
        ),
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@configclass
class G1GoalObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=vel_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=vel_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=vel_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 目标点（相对位姿）作为唯一的"任务输入"
        pose_command = ObsTerm(
            func=nav_mdp.generated_commands,
            params={"command_name": "pose_command"},
        )
        joint_pos = ObsTerm(
            func=vel_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=vel_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=vel_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
@configclass
class G1GoalRewardsCfg(G1Rewards):
    """继承 G1Rewards，复用所有"行走稳定性"相关 reward；
    覆盖 feet_air_time 为 pose-based 版本；新增导航相关 reward。"""

    # ----- 覆盖：用 pose-based 版本替换原 velocity-based feet_air_time -----
    feet_air_time = RewTerm(
        func=custom_rewards.feet_air_time_positive_biped_pose,
        weight=0.75,  # 与 G1FlatEnvCfg 一致
        params={
            "command_name": "pose_command",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "threshold": 0.4,
            "distance_threshold": 0.2,
        },
    )

    # ----- 新增：导航 task reward -----
    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=2.0,
        params={"std": 1.5, "command_name": "pose_command"},
    )
    position_tracking_fine = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.25, "command_name": "pose_command"},
    )
    heading_tracking = RewTerm(
        func=nav_mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )
    goal_reached = RewTerm(
        func=custom_rewards.goal_reached_bonus,
        weight=0.5,
        params={"command_name": "pose_command", "threshold": 0.3},
    )
    stand_still_at_goal = RewTerm(
        func=custom_rewards.stand_still_at_goal,
        weight=1.0,
        params={"command_name": "pose_command", "threshold": 0.3},
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@configclass
class G1GoalCfg(G1RoughEnvCfg):
    commands: G1GoalCommandsCfg = G1GoalCommandsCfg()
    observations: G1GoalObservationsCfg = G1GoalObservationsCfg()
    rewards: G1GoalRewardsCfg = G1GoalRewardsCfg()

    def __post_init__(self):
        # 父类会访问 self.commands.base_velocity.ranges.*，所以必须先调用
        super().__post_init__()

        # ---------- Flat terrain ----------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None  # 不要课程学习

        # ---------- 关闭 velocity command ----------
        # 注意：G1Rewards 里的 track_lin_vel_xy_exp / track_ang_vel_z_exp
        # 是在 G1Rewards.__init__ 之前从 RewardsCfg 继承下来的。
        # 现在 base_velocity 被关掉了，所以这两个 reward 也必须关掉。
        self.commands.base_velocity = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        # 父类 RewardsCfg 里的 undesired_contacts 用的是 .*THIGH（ANYmal 命名），
        # G1 上原本就被 G1RoughEnvCfg 关掉了，这里再确认一下
        self.rewards.undesired_contacts = None

        # ---------- 行走稳定性 reward 微调 ----------
        # 这些权重沿用 G1FlatEnvCfg 的设置，对 flat 地形是验证过的
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005

        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # 步态滑动 / 关节限位
        self.rewards.feet_slide.weight = -0.1
        self.rewards.dof_pos_limits.weight = -1.0

        # ---------- 终止条件 ----------
        # 仅 timeout + torso 接触地面终止；到达目标不终止（让 dense reward 主导）
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

        # ---------- 起始位姿和目标点范围已在 events / commands 中设置 ----------
        # （父类已配置 reset_base，这里不重复）


# ---------------------------------------------------------------------------
# PLAY config
# ---------------------------------------------------------------------------
@configclass
class G1GoalCfg_PLAY(G1GoalCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # 关闭观测噪声
        self.observations.policy.enable_corruption = False

        # 关闭随机扰动
        self.events.base_external_force_torque = None
        self.events.push_robot = None
