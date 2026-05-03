# modules/buffers/simple_trajectory_buffer.py

import torch
from dataclasses import dataclass


@dataclass
class TrajectoryBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    logp: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor


class SimpleTrajectoryBuffer:
    def __init__(
        self,
        max_traj_num,
        max_traj_len,
        obs_dim,
        action_dim,
        device,
    ):

        self.max_traj_num = max_traj_num
        self.max_traj_len = max_traj_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.obs = torch.zeros(max_traj_num, max_traj_len, obs_dim, device=device)
        self.action = torch.zeros(max_traj_num, max_traj_len, action_dim, device=device)
        self.reward = torch.zeros(max_traj_num, max_traj_len, 1, device=device)
        self.next_obs = torch.zeros(max_traj_num, max_traj_len, obs_dim, device=device)
        self.done = torch.zeros(max_traj_num, max_traj_len, 1, device=device)
        self.logp = torch.zeros(max_traj_num, max_traj_len, 1, device=device)
        self.mask = torch.zeros(max_traj_num, max_traj_len, 1, device=device)

        self.lengths = torch.zeros(max_traj_num, device=device, dtype=torch.long)

        self.ptr = 0
        self.size = 0

    def reset(self):
        self.ptr = 0
        self.size = 0
        self.mask.zero_()
        self.lengths.zero_()
        self.obs.zero_()
        self.action.zero_()
        self.reward.zero_()
        self.next_obs.zero_()
        self.done.zero_()
        self.logp.zero_()

    def __len__(self):
        return self.size

    def _to_tensor(self, x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
            if dtype is not None:
                x = x.to(dtype)
            return x
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _ensure_3d_last_one(self, x):
        """
        把 reward / done / logp 变成 [B, T, 1]
        """

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x

    def add_rollout(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        logp,
        lengths=None,
        mask=None,
    ):
        obs = self._to_tensor(obs, dtype=torch.float32)
        action = self._to_tensor(action, dtype=torch.float32)
        reward = self._ensure_3d_last_one(self._to_tensor(reward, dtype=torch.float32))
        next_obs = self._to_tensor(next_obs, dtype=torch.float32)
        done = self._ensure_3d_last_one(self._to_tensor(done, dtype=torch.float32))
        logp = self._ensure_3d_last_one(self._to_tensor(logp, dtype=torch.float32))

        num_trajs, traj_len = obs.shape[0], obs.shape[1]

        if traj_len > self.max_traj_len:
            raise RuntimeError(
                f"traj_len={traj_len} is larger than max_traj_len={self.max_traj_len}"
            )
        start = self.ptr
        end = self.ptr + num_trajs

        self.obs[start:end, :traj_len] = obs
        self.action[start:end, :traj_len] = action
        self.reward[start:end, :traj_len] = reward
        self.next_obs[start:end, :traj_len] = next_obs
        self.done[start:end, :traj_len] = done
        self.logp[start:end, :traj_len] = logp

        if lengths is not None:
            lengths = self._to_tensor(lengths, dtype=torch.long)
            self.lengths[start:end] = lengths

        if mask is not None:
            mask = self._ensure_3d_last_one(self._to_tensor(mask, dtype=torch.float32))
            self.mask[start:end, :traj_len] = mask

            if lengths is None:
                inferred_lengths = mask.squeeze(-1).sum(dim=1).long()
                self.lengths[start:end] = inferred_lengths

        else:
            if lengths is None:
                raise RuntimeError("Either lengths or mask must be provided.")

            time_ids = torch.arange(
                traj_len,
                device=self.device,
            ).view(1, traj_len)

            valid = time_ids < lengths.view(num_trajs, 1)
            self.mask[start:end, :traj_len, 0] = valid.float()

        self.ptr += num_trajs
        self.size += num_trajs

    def get_all(self) -> TrajectoryBatch:
        return TrajectoryBatch(
            obs=self.obs[: self.size],
            action=self.action[: self.size],
            reward=self.reward[: self.size],
            next_obs=self.next_obs[: self.size],
            done=self.done[: self.size],
            logp=self.logp[: self.size],
            mask=self.mask[: self.size],
            lengths=self.lengths[: self.size],
        )

    def sample(
        self,
        batch_size,
        shuffle=True,
        drop_last=False,
    ):
        if shuffle:
            indices = torch.randperm(self.size, device=self.device)
        else:
            indices = torch.arange(self.size, device=self.device)

        for start in range(0, self.size, batch_size):
            end = start + batch_size

            if end > self.size and drop_last:
                break

            batch_idx = indices[start:end]

            yield self._make_batch(batch_idx)

    def sample_one_batch(self, batch_size):
        batch_size = min(batch_size, self.size)
        batch_idx = torch.randperm(self.size, device=self.device)[:batch_size]
        return self._make_batch(batch_idx)

    def _make_batch(self, batch_idx):
        return TrajectoryBatch(
            obs=self.obs[batch_idx],
            action=self.action[batch_idx],
            reward=self.reward[batch_idx],
            next_obs=self.next_obs[batch_idx],
            done=self.done[batch_idx],
            logp=self.logp[batch_idx],
            mask=self.mask[batch_idx],
            lengths=self.lengths[batch_idx],
        )
