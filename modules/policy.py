import torch
from torch import nn
from .mlp import MLP
import math

LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.0


class Policy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dims, activations, output_activation
    ):
        super(Policy, self).__init__()
        self.mlp = MLP(obs_dim, action_dim, hidden_dims, activations, output_activation)
        init_log_std = 0.0
        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32)
        )

    def set_sample_std(self, std):
        std = torch.as_tensor(std, dtype=torch.float32, device=self.log_std.device)
        if std.ndim == 0:
            std = std.expand_as(self.log_std)
        with torch.no_grad():
            self.log_std.copy_(torch.log(std))

    def get_sample_std(self):
        return torch.exp(self.log_std)

    def _get_dist(self, mean):
        log_std = torch.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def logp(self, mean, action):
        dist = self._get_dist(mean)
        return dist.log_prob(action).sum(dim=-1, keepdim=True)

    def forward(self, obs):
        mean = self.mlp(obs)
        dist = self._get_dist(mean)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return mean, action, logp

    def evaluate_actions(self, obs, actions):
        action_mean = self.mlp(obs)
        dist = self._get_dist(action_mean)

        logp = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action_mean, logp, entropy
