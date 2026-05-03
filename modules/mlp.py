import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dims, activations, output_activation
    ):
        super(MLP, self).__init__()
        assert len(hidden_dims) == len(
            activations
        ), "Length of hidden_dims and activations must be the same"
        activation_dict = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "identity": nn.Identity,
        }
        layers = []
        prev_dim = input_dim
        for hidden_dim, activation in zip(hidden_dims, activations):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_dict[activation]())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(activation_dict[output_activation]())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
