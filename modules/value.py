import torch
from torch import nn
from .mlp import MLP


class Value(nn.Module):
    def __init__(self, obs_dim, hidden_dims, activations, output_activation):
        super(Value, self).__init__()
        self.mlp = MLP(obs_dim, 1, hidden_dims, activations, None)

    def forward(self, obs):
        return self.mlp(obs)
