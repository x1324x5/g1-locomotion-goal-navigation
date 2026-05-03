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

from .mdp import rewards as custom_rewards


@configclass
class G1GoalCommandsCfg(VelocityCommandsCfg):
    pose_command = nav_mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=nav_mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            heading=(-math.pi, math.pi),
        ),
    )


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


@configclass
class G1GoalRewardsCfg(G1Rewards):
    feet_air_time = RewTerm(
        func=custom_rewards.feet_air_time_positive_biped_pose,
        weight=0.75,
        params={
            "command_name": "pose_command",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "threshold": 0.4,
            "distance_threshold": 0.2,
        },
    )

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


@configclass
class G1GoalCfg(G1RoughEnvCfg):
    commands: G1GoalCommandsCfg = G1GoalCommandsCfg()
    observations: G1GoalObservationsCfg = G1GoalObservationsCfg()
    rewards: G1GoalRewardsCfg = G1GoalRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        self.rewards.undesired_contacts = None
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

        self.rewards.feet_slide.weight = -0.1
        self.rewards.dof_pos_limits.weight = -1.0

        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1GoalCfg_PLAY(G1GoalCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
