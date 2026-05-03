import os
import time
import random
import logging
from collections import deque

import numpy as np
import torch

from .Base_alg import BaseAlg
from modules.policy import Policy

logger = logging.getLogger("unrl.ppo_play")


class PPOPlay(BaseAlg):
    def __init__(self, parameter, env=None):
        super(PPOPlay, self).__init__(parameter, env)

        self.parameter = parameter
        self.device = parameter.device
        self.obs_dim = parameter.obs_dim
        self.action_dim = parameter.action_dim
        self.num_envs = parameter.num_envs
        self.env_name = parameter.task

        self.load_model_dir = parameter.load_model_dir
        self.deterministic = getattr(parameter, "deterministic", False)
        self.play_steps = getattr(parameter, "play_steps", -1)
        self.log_interval = getattr(parameter, "log_interval", 1000)

        self._seed(parameter.seed)

        self.policy = self.load_policy(self.load_model_dir)
        self.policy.eval()

        self.obs = None
        self.action = torch.zeros(
            (self.num_envs, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.current_episode_return = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.current_episode_step = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.long,
            device=self.device,
        )

        self.return_window = deque(maxlen=1000)
        self.length_window = deque(maxlen=1000)

        self.total_env_steps = 0
        self.episode_nums = 0

        self.env_reset()

        logger.info("[PPOPlay] Initialized")

    def _seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _to_tensor(self, x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _to_column(self, x, dtype=torch.float32):
        x = self._to_tensor(x, dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        return x

    def _get_policy_obs(self, obs):
        """
        兼容两种 obs 格式：

        1. obs 是 dict:
            obs["policy"]

        2. obs 直接就是 tensor / ndarray
        """
        if isinstance(obs, dict):
            obs = obs["policy"]
        return self._to_tensor(obs, dtype=torch.float32)

    def env_reset(self, reset_buf=None):
        """
        reset_buf != None 时只重置 play 侧的 episode 统计。

        注意：
        这里假设你的环境和训练时一样，是 auto-reset vector env。
        也就是 env.step() 在 done 后返回的 next_obs 已经是下一个 episode 的 obs。
        """
        if reset_buf is not None:
            reset_buf = reset_buf.squeeze(-1).bool()
            self.current_episode_return[reset_buf] = 0.0
            self.current_episode_step[reset_buf] = 0
            return

        obs, _ = self.env.reset()
        self.obs = self._get_policy_obs(obs)

        self.current_episode_return.zero_()
        self.current_episode_step.zero_()

    def load_policy(self, model_dir):
        if model_dir is None:
            raise RuntimeError("load_model_dir is None. Please set it in play config.")

        policy_path = os.path.join(model_dir, "policy.pt")
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

        policy = Policy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.parameter.policy_hidden_dims,
            activations=self.parameter.policy_activations,
            output_activation=self.parameter.policy_output_activation,
        ).to(self.device)

        state_dict = torch.load(policy_path, map_location=self.device)

        policy.load_state_dict(state_dict)
        policy.eval()

        logger.info(f"[PPOPlay] Policy loaded from {policy_path}")
        logger.info(
            f"Policy sample std: {policy.get_sample_std().detach().cpu().numpy()}"
        )
        return policy

    @torch.no_grad()
    def select_action(self, obs):
        action_mean, action_sample, _ = self.policy.forward(obs)

        if self.deterministic:
            return action_mean
        else:
            return action_sample

    @torch.no_grad()
    def play(self):
        step = 0

        while self.play_steps < 0 or step < self.play_steps:
            self.action = self.select_action(self.obs)

            next_obs, reward, terminated, truncated, info = self.env.step(self.action)

            next_obs = self._get_policy_obs(next_obs)
            reward = self._to_column(reward, dtype=torch.float32)
            terminated = self._to_column(terminated, dtype=torch.bool)
            truncated = self._to_column(truncated, dtype=torch.bool)

            done = terminated | truncated

            self.current_episode_return += reward
            self.current_episode_step += 1

            self.obs = next_obs
            self.total_env_steps += self.num_envs
            step += 1

            if done.any():
                done_squeeze = done.squeeze(-1)

                ep_returns = self.current_episode_return[done_squeeze]
                ep_lengths = self.current_episode_step[done_squeeze]

                self.return_window.extend(ep_returns.cpu().numpy().reshape(-1).tolist())
                self.length_window.extend(ep_lengths.cpu().numpy().reshape(-1).tolist())

                self.episode_nums += int(done_squeeze.sum().item())

                logger.info(
                    "[PPOPlay] "
                    f"episodes={self.episode_nums}, "
                    f"return_mean={float(ep_returns.mean().item()):.2f}, "
                    f"length_mean={float(ep_lengths.float().mean().item()):.2f}, "
                    f"window_return_mean={float(np.mean(self.return_window)):.2f}"
                )

                self.env_reset(done)

            if self.log_interval > 0 and step % self.log_interval == 0:
                if len(self.return_window) > 0:
                    logger.info(
                        "[PPOPlay] "
                        f"steps={self.total_env_steps}, "
                        f"episodes={self.episode_nums}, "
                        f"return_window_mean={float(np.mean(self.return_window)):.2f}, "
                        f"length_window_mean={float(np.mean(self.length_window)):.2f}"
                    )
                else:
                    logger.info(
                        "[PPOPlay] "
                        f"steps={self.total_env_steps}, "
                        f"episodes={self.episode_nums}"
                    )

    def train(self):
        self.play()
