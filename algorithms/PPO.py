import os
import torch
from torch import nn
import numpy as np
import random
import time
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import yaml
import logging
from tqdm import tqdm
from .Base_alg import BaseAlg
from modules.policy import Policy
from modules.value import Value
from buffers.simple_buffer import SimpleTrajectoryBuffer
from utility.ppo_utils import compute_gae, n2t

logger = logging.getLogger(__name__)


class PPO(BaseAlg):
    def __init__(self, parameter, env=None):
        super(PPO, self).__init__(parameter, env)
        self.device = self.parameter.device
        self.obs_dim = self.parameter.obs_dim
        self.action_dim = self.parameter.action_dim
        self.total_iterations = self.parameter.total_iterations
        self.max_episode_steps = self.parameter.max_episode_steps
        self.rollout_steps = self.parameter.rollout_steps
        self.num_envs = self.parameter.num_envs
        self.env_name = parameter.task
        self._seed(self.parameter.seed)
        self.make_policy()
        self.make_value()
        self.optim_class = torch.optim.AdamW
        self.policy_optimizer = self.optim_class(
            self.policy.parameters(),
            lr=self.parameter.policy_lr,
            weight_decay=self.parameter.policy_weight_decay,
        )
        self.value_optimizer = self.optim_class(
            self.value.parameters(),
            lr=self.parameter.value_lr,
            weight_decay=self.parameter.value_weight_decay,
        )
        self.policy.set_sample_std(self.parameter.policy_sample_std)
        self.entropy_coef = self.parameter.entropy_coef
        frac = lambda step: 1.0 - step / self.total_iterations
        self.policy_sched = LambdaLR(self.policy_optimizer, frac)
        self.value_sched = LambdaLR(self.value_optimizer, frac)
        self.replay_buffer = SimpleTrajectoryBuffer(
            max_traj_num=int(
                self.parameter.num_envs
                * 32
                * self.rollout_steps
                / self.max_episode_steps
            ),
            max_traj_len=self.rollout_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device="cpu",
        )

        self.iteraion = 0
        self.grad_num = 0
        self.sample_num = 0
        self.total_env_steps = 0
        self.episode_nums = 0

        self.return_window = deque(maxlen=1000)
        self.length_window = deque(maxlen=1000)

        self.obs = None
        self.action = None
        self.reward = None
        self.next_obs = None
        self.done = None
        self.last_action = None
        self.last_obs = None
        self.last_reward = None
        self.current_episode_step = torch.zeros(
            (self.num_envs, 1), dtype=torch.long, device=self.device
        )
        self.current_episode_return = torch.zeros(
            (self.num_envs, 1), dtype=torch.float, device=self.device
        )
        self.env_reset()

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        output_dir = os.path.join(
            self.parameter.output_dir,
            f"{self.env_name}_PPO",
            date_str,
            time_str,
        )
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        with open(os.path.join(self.output_dir, "parameter.yaml"), "w") as f:
            yaml.dump(vars(parameter), f)

        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        self.debug = self.parameter.debug
        self.load_model_dir = self.parameter.load_model_dir
        logger.info(f"Initialized PPO")

    def _seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def make_policy(self):
        self.policy: Policy = Policy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.parameter.policy_hidden_dims,
            activations=self.parameter.policy_activations,
            output_activation=self.parameter.policy_output_activation,
        ).to(self.device)

    def make_value(self):
        self.value: Value = Value(
            obs_dim=self.obs_dim,
            hidden_dims=self.parameter.value_hidden_dims,
            activations=self.parameter.value_activations,
            output_activation=self.parameter.value_output_activation,
        ).to(self.device)

    def env_reset(self, reset_buf=None):
        if reset_buf is not None:
            reset_buf = reset_buf.squeeze(-1).bool()
            self.last_obs[reset_buf] = 0.0
            self.last_action[reset_buf] = 0.0
            self.last_reward[reset_buf] = 0.0
            self.action[reset_buf] = 0.0
            self.reward[reset_buf] = 0.0
            self.next_start_flag[reset_buf] = 1.0
            return
        obs, _ = self.env.reset()
        self.obs = obs["policy"]
        self.last_obs = torch.zeros_like(self.obs)
        self.action = torch.zeros(
            (self.num_envs, self.action_dim), dtype=torch.float, device=self.device
        )
        self.last_action = torch.zeros_like(self.action)
        self.reward = torch.zeros(
            (self.num_envs, 1), dtype=torch.float, device=self.device
        )
        self.last_reward = torch.zeros_like(self.reward)
        self.next_start_flag = torch.ones(
            (self.num_envs, 1), dtype=torch.float, device=self.device
        )
        self.current_episode_return.zero_()
        self.current_episode_step.zero_()

    def _to_column(self, x):
        if len(x.shape) == 1:
            return x.unsqueeze(-1)
        return x

    @torch.no_grad()
    def collect_rollout(self):
        start_time = time.time()
        self.policy.eval()
        T = self.rollout_steps
        N = self.num_envs
        S = self.obs_dim
        A = self.action_dim

        obs_buf = torch.zeros((N, T, S), device=self.device)
        last_obs_buf = torch.zeros((N, T, S), device=self.device)
        next_obs_buf = torch.zeros((N, T, S), device=self.device)
        action_buf = torch.zeros((N, T, A), device=self.device)
        last_action_buf = torch.zeros((N, T, A), device=self.device)
        reward_buf = torch.zeros((N, T, 1), device=self.device)
        last_reward_buf = torch.zeros((N, T, 1), device=self.device)
        start_flag_buf = torch.zeros((N, T, 1), device=self.device)
        logp_buf = torch.zeros((N, T, 1), device=self.device)
        term_buf = torch.zeros((N, T, 1), device=self.device)
        trunc_buf = torch.zeros((N, T, 1), device=self.device)
        mask_buf = torch.zeros((N, T, 1), device=self.device)
        done_buf = torch.zeros((N, T, 1), device=self.device)
        start_buf = torch.zeros((N, T, 1), device=self.device)

        ep_returns = []
        ep_lengths = []
        current_done_episode = 0
        for step in tqdm(range(T)):
            action_mean, action_sample, logp = self.policy.forward(self.obs)
            self.action = action_sample
            next_obs, reward, terminated, truncated, info = self.env.step(self.action)

            next_obs = next_obs["policy"]
            reward = self._to_column(reward)
            terminated = self._to_column(terminated)
            truncated = self._to_column(truncated)
            done = terminated | truncated
            self.reward = reward.clone()

            self.current_episode_return += reward
            self.current_episode_step += 1

            obs_buf[:, step] = self.obs
            last_obs_buf[:, step] = self.last_obs
            next_obs_buf[:, step] = next_obs
            action_buf[:, step] = self.action
            last_action_buf[:, step] = self.last_action
            reward_buf[:, step] = self.reward
            last_reward_buf[:, step] = self.last_reward
            start_flag_buf[:, step] = self.next_start_flag
            logp_buf[:, step] = self._to_column(logp)
            term_buf[:, step] = terminated
            trunc_buf[:, step] = truncated
            done_buf[:, step] = done
            start_buf[:, step] = self.next_start_flag

            self.last_obs = self.obs
            self.last_action = self.action
            self.last_reward = self.reward
            self.obs = next_obs.clone()
            self.next_start_flag.zero_()
            if done.any():
                done_squeeze = done.squeeze(-1)
                ep_returns.extend(
                    self.current_episode_return[done_squeeze].cpu().numpy().tolist()
                )
                ep_lengths.extend(
                    self.current_episode_step[done_squeeze].cpu().numpy().tolist()
                )
                self.return_window.extend(
                    self.current_episode_return[done_squeeze].cpu().numpy().tolist()
                )
                self.length_window.extend(
                    self.current_episode_step[done_squeeze].cpu().numpy().tolist()
                )
                self.episode_nums += done.sum().item()
                self.current_episode_return[done_squeeze] = 0.0
                self.current_episode_step[done_squeeze] = 0
                self.env_reset(done)

            self.total_env_steps += self.num_envs

        traj_buffer = torch.cat(
            [
                obs_buf,
                last_obs_buf,
                next_obs_buf,
                action_buf,
                last_action_buf,
                reward_buf,
                last_reward_buf,
                logp_buf,
                term_buf,
                trunc_buf,
                mask_buf,
                done_buf,
                start_buf,
            ],
            dim=-1,
        )
        end_time = time.time()
        return {
            "traj_buffer": traj_buffer,
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
            "time_cost": end_time - start_time,
        }

    @torch.no_grad()
    def add_rollout(self, traj_buffer):
        if not isinstance(traj_buffer, torch.Tensor):
            traj_buffer = torch.as_tensor(traj_buffer, dtype=torch.float32)
        else:
            traj_buffer = traj_buffer
        work_device = torch.device("cpu")
        S, A = self.obs_dim, self.action_dim
        expected_dim = 3 * S + 2 * A + 8
        if traj_buffer.shape[-1] != expected_dim:
            raise RuntimeError(
                f"traj_buffer feature dim mismatch: "
                f"got {traj_buffer.shape[-1]}, expected {expected_dim}"
            )

        N, T = traj_buffer.shape[:2]

        def _slice(start, dim):
            return traj_buffer[..., start : start + dim]

        obs = _slice(0, S)
        next_obs = _slice(2 * S, S)
        action = _slice(3 * S, A)
        reward = _slice(3 * S + 2 * A, 1)
        logp = _slice(3 * S + 2 * A + 2, 1)
        terminated = _slice(3 * S + 2 * A + 3, 1)
        truncated = _slice(3 * S + 2 * A + 4, 1)

        boundary = ((terminated > 0.5) | (truncated > 0.5)).squeeze(-1)  # (N, T) bool

        boundary = boundary.clone()
        boundary[:, -1] = True

        prev_boundary = torch.zeros_like(boundary, dtype=torch.long)
        prev_boundary[:, 1:] = boundary[:, :-1].long()
        traj_id_in_env = prev_boundary.cumsum(dim=1)

        num_trajs_per_env = traj_id_in_env[:, -1] + 1
        env_offsets = torch.zeros(N, dtype=torch.long, device=self.device)
        env_offsets[1:] = num_trajs_per_env[:-1].cumsum(dim=0)
        global_traj_id = traj_id_in_env + env_offsets.view(N, 1)
        total_trajs = int(num_trajs_per_env.sum().item())
        t_in_env = torch.arange(T, device=self.device).expand(N, T)
        is_traj_start = torch.zeros_like(boundary)
        is_traj_start[:, 0] = True
        is_traj_start[:, 1:] = boundary[:, :-1]
        t_start = torch.where(is_traj_start, t_in_env, torch.full_like(t_in_env, -1))
        traj_start_t, _ = t_start.cummax(dim=1)
        t_within_traj = t_in_env - traj_start_t
        lengths = torch.zeros(total_trajs, dtype=torch.long, device=self.device)
        lengths.scatter_add_(
            0,
            global_traj_id.reshape(-1),
            torch.ones(N * T, dtype=torch.long, device=self.device),
        )
        max_len = self.rollout_steps
        g_flat = global_traj_id.reshape(-1)
        t_flat = t_within_traj.reshape(-1)

        def pack(field):
            D = field.shape[-1]
            out = torch.zeros(
                total_trajs, max_len, D, device=self.device, dtype=field.dtype
            )
            out[g_flat, t_flat] = field.reshape(-1, D)
            return out

        obs_p = pack(obs)
        action_p = pack(action)
        reward_p = pack(reward)
        next_obs_p = pack(next_obs)
        logp_p = pack(logp)
        done_p = pack(terminated.float())

        time_ids = torch.arange(max_len, device=self.device).view(1, max_len)
        valid_2d = time_ids < lengths.view(total_trajs, 1)
        mask = valid_2d.unsqueeze(-1).float()

        keep = lengths > 1
        if not bool(keep.all()):
            keep_idx = keep.nonzero(as_tuple=True)[0]
            obs_p, action_p, reward_p = (
                obs_p[keep_idx],
                action_p[keep_idx],
                reward_p[keep_idx],
            )
            next_obs_p, logp_p, done_p = (
                next_obs_p[keep_idx],
                logp_p[keep_idx],
                done_p[keep_idx],
            )
            mask, lengths = mask[keep_idx], lengths[keep_idx]
            total_trajs = lengths.shape[0]
        logger.info(
            f"Added {total_trajs} trajs, {lengths.sum().item()} in iteration {self.iteraion}"
        )
        if total_trajs == 0:
            return

        if lengths.max().item() > self.replay_buffer.max_traj_len:
            raise RuntimeError(
                f"max parsed trajectory length {lengths.max().item()} exceeds "
                f"replay_buffer.max_traj_len={self.replay_buffer.max_traj_len}. "
                f"Set max_traj_len >= rollout_steps."
            )
        if total_trajs > self.replay_buffer.max_traj_num:
            raise RuntimeError(
                f"parsed {total_trajs} trajectories, but "
                f"replay_buffer.max_traj_num={self.replay_buffer.max_traj_num}. "
                f"Increase max_traj_num."
            )
        self.replay_buffer.reset()
        self.replay_buffer.add_rollout(
            obs=obs_p,
            action=action_p,
            reward=reward_p,
            next_obs=next_obs_p,
            done=done_p,
            logp=logp_p,
            lengths=lengths,
            mask=mask,
        )

        if self.debug:
            logger.info(
                f"[PPO] Added {int(lengths.sum().item())} transitions, "
                f"{total_trajs} trajs. "
                f"Traj lengths: max={int(lengths.max().item())}, "
                f"min={int(lengths.min().item())}"
            )

    def get_advantages_and_returns(self, batch):
        obs = batch.obs
        next_obs = batch.next_obs
        reward = batch.reward
        done = batch.done
        valid_mask = n2t(batch.mask, self.device)

        with torch.no_grad():
            values = self.value(n2t(obs, self.device))
            next_values = self.value(n2t(next_obs, self.device))
            advantages, returns = compute_gae(
                rewards=n2t(reward, self.device),
                values=n2t(values, self.device),
                next_values=n2t(next_values, self.device),
                done=n2t(done, self.device),
                gamma=self.parameter.gamma,
                lam=self.parameter.lam,
                mask=valid_mask,
            )
            advantages = advantages * n2t(batch.mask, self.device)
            returns = returns * n2t(batch.mask, self.device)
            valid_adv = advantages[valid_mask.bool()]

            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std(unbiased=False)

            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = advantages * valid_mask
        return advantages, returns

    def update(self):
        batch = self.replay_buffer.get_all()
        self.policy.train()
        self.value.train()

        logs = {}
        advantages, returns = self.get_advantages_and_returns(batch)
        indices = np.arange(batch.obs.shape[0])
        mini_batch_size = self.parameter.batch_size
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fracs = []
        ratio_means = []
        ratio_mins = []
        ratio_maxs = []
        ppo_start = time.time()
        for epoch in tqdm(range(self.parameter.ppo_epochs)):
            np.random.shuffle(indices)
            for start in range(0, batch.obs.shape[0], mini_batch_size):
                end = min(start + mini_batch_size, batch.obs.shape[0])
                mb_idx = indices[start:end]
                mb_obs = n2t(batch.obs[mb_idx], self.device)
                mb_action = n2t(batch.action[mb_idx], self.device)
                mb_logp_old = n2t(batch.logp[mb_idx], self.device)
                mb_advantages = n2t(advantages[mb_idx], self.device)
                mb_returns = n2t(returns[mb_idx], self.device)
                mb_valid_mask = n2t(batch.mask[mb_idx], self.device)
                action_mean, logp, entropy = self.policy.evaluate_actions(
                    mb_obs, mb_action
                )
                new_values = self.value(mb_obs)
                log_ratio = logp - mb_logp_old
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_advantages
                policy_loss = -torch.min(surr1, surr2) * mb_valid_mask
                policy_loss = policy_loss.sum() / mb_valid_mask.sum()
                value_loss = (new_values - mb_returns) ** 2 * mb_valid_mask
                value_loss = value_loss.sum() / mb_valid_mask.sum()
                entropy_loss = -entropy * mb_valid_mask
                entropy_loss = entropy_loss.sum() / mb_valid_mask.sum()
                total_loss = (
                    policy_loss
                    + self.parameter.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.parameter.max_grad_norm,
                )
                self.policy_optimizer.step()
                self.value_optimizer.step()
                with torch.no_grad():
                    valid = mb_valid_mask.bool()

                    approx_kl = (ratio - 1.0) - log_ratio
                    approx_kl = approx_kl[valid].mean()
                    clip_frac = (torch.abs(ratio - 1.0) > 0.2).float() * mb_valid_mask
                    clip_frac = clip_frac.sum() / mb_valid_mask.sum().clamp_min(1.0)

                    valid_ratio = ratio[valid]
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_valid_mean = (
                    entropy * mb_valid_mask
                ).sum() / mb_valid_mask.sum()
                entropy_losses.append(entropy_valid_mean.item())
                total_losses.append(total_loss.item())
                approx_kls.append(approx_kl.item())
                clip_fracs.append(clip_frac.item())
                ratio_means.append(valid_ratio.mean().item())
                ratio_mins.append(valid_ratio.min().item())
                ratio_maxs.append(valid_ratio.max().item())
        self.policy_sched.step()
        self.value_sched.step()
        ppo_time = time.time() - ppo_start
        valid_mask = batch.mask.bool()

        logs["loss/policy"] = float(np.mean(policy_losses))
        logs["loss/value"] = float(np.mean(value_losses))
        logs["loss/entropy"] = float(np.mean(entropy_losses))
        logs["loss/total"] = float(np.mean(total_losses))
        logs["ppo/clip_frac"] = float(np.mean(clip_fracs))
        logs["ppo/ratio_mean"] = float(np.mean(ratio_means))
        logs["ppo/ratio_min"] = float(np.mean(ratio_mins))
        logs["ppo/ratio_max"] = float(np.mean(ratio_maxs))

        logs["ppo/approx_kl"] = float(np.mean(approx_kls))
        logs["train/ppo_update_time"] = ppo_time
        logs["train/policy_lr"] = self.policy_optimizer.param_groups[0]["lr"]
        logs["train/value_lr"] = self.value_optimizer.param_groups[0]["lr"]

        logs["advantage/mean"] = advantages[valid_mask].mean().item()
        logs["advantage/std"] = advantages[valid_mask].std().item()
        logs["return/mean"] = returns[valid_mask].mean().item()
        logs["return/std"] = returns[valid_mask].std().item()

        return logs

    def write_logs(self, rollout_info, update_logs):
        step = self.total_env_steps

        ep_returns = rollout_info["ep_returns"]
        ep_lengths = rollout_info["ep_lengths"]

        if len(ep_returns) > 0:
            self.writer.add_scalar(
                "rollout/episode_return_mean",
                float(np.mean(ep_returns)),
                step,
            )
            self.writer.add_scalar(
                "rollout/episode_return_max",
                float(np.max(ep_returns)),
                step,
            )
            self.writer.add_scalar(
                "rollout/episode_return_min",
                float(np.min(ep_returns)),
                step,
            )

        if len(ep_lengths) > 0:
            self.writer.add_scalar(
                "rollout/episode_length_mean",
                float(np.mean(ep_lengths)),
                step,
            )

        if len(self.return_window) > 0:
            self.writer.add_scalar(
                "rollout/return_window_mean",
                float(np.mean(self.return_window)),
                step,
            )

        if len(self.length_window) > 0:
            self.writer.add_scalar(
                "rollout/length_window_mean",
                float(np.mean(self.length_window)),
                step,
            )

        self.writer.add_scalar(
            "rollout/num_finished_episodes",
            len(ep_returns),
            step,
        )
        self.writer.add_scalar(
            "rollout/time_cost",
            rollout_info["time_cost"],
            step,
        )

        self.writer.add_scalar(
            "train/iteration",
            self.iteraion,
            step,
        )
        self.writer.add_scalar(
            "train/total_env_steps",
            self.total_env_steps,
            step,
        )
        self.writer.add_scalar(
            "train/episode_nums",
            self.episode_nums,
            step,
        )
        self.writer.add_scalar(
            "policy/sample_std",
            self.policy.get_sample_std().mean(),
            step,
        )
        self.writer.add_scalar(
            "policy/sample_std_min", self.policy.get_sample_std().min().item(), step
        )
        self.writer.add_scalar(
            "policy/sample_std_max", self.policy.get_sample_std().max().item(), step
        )
        self.writer.add_scalar(
            "policy/log_std_mean", self.policy.log_std.mean().item(), step
        )

        for key, value in update_logs.items():
            self.writer.add_scalar(key, value, step)

        self.writer.flush()

    def train(self):
        self.load(self.load_model_dir, load_policy=True, load_value=True)
        for iteration in range(self.total_iterations):
            self.iteraion = iteration
            rollout_info = self.collect_rollout()
            logger.info(
                f"[PPO] Iteration {iteration}: Rollout {rollout_info['time_cost']}s, "
                f"Average Return {np.mean(rollout_info['ep_returns']):.2f}, "
                f"Average Length {np.mean(rollout_info['ep_lengths']):.2f}, "
                f"Episodes {len(rollout_info['ep_lengths'])}"
            )
            self.add_rollout(rollout_info["traj_buffer"])
            update_logs = self.update()
            self.write_logs(rollout_info, update_logs)
            if (iteration + 1) % self.parameter.save_interval == 0:
                self.save()

    def save(self, model_dir=None):
        if model_dir is None:
            ret_str = ""
            if len(self.return_window) > 0:
                ret_str = f"_reward{float(np.mean(self.return_window)):.1f}"
            model_dir = os.path.join(
                self.output_dir, "model", f"{ret_str}_{self.iteraion}"
            )
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(model_dir, "policy.pt"))
        torch.save(self.value.state_dict(), os.path.join(model_dir, "value.pt"))

        logger.info(f"[PPO] Model saved to {model_dir}")

    def load(self, model_dir=None, load_policy=True, load_value=True):
        if model_dir is None:
            return
        if load_policy:
            policy_path = os.path.join(model_dir, "policy.pt")
            self.policy.load_state_dict(
                torch.load(policy_path, map_location=self.device)
            )
            logger.info(f"[PPO] Policy loaded from {policy_path}")
        if load_value:
            value_path = os.path.join(model_dir, "value.pt")
            self.value.load_state_dict(torch.load(value_path, map_location=self.device))
            logger.info(f"[PPO] Value loaded from {value_path}")
