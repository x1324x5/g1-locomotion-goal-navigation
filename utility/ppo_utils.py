import numpy as np
import torch
from torch import nn
import logging


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    done: torch.Tensor,
    gamma,
    lam,
    mask: torch.Tensor,
):

    gae = torch.zeros_like(rewards[:, 0])
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    T = rewards.shape[1]

    for step in reversed(range(T)):
        nonterminal = 1.0 - done[:, step].float()
        delta = (
            rewards[:, step]
            + gamma * next_values[:, step] * nonterminal
            - values[:, step]
        )
        gae = delta + gamma * lam * nonterminal * gae
        gae = gae * mask[:, step]
        advantages[:, step] = gae
        returns[:, step] = (gae + values[:, step]) * mask[:, step]
    return advantages, returns


def n2t(data, device):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device, dtype=torch.float32)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype=torch.float32)
    elif isinstance(data, list):
        return [n2t(item, device) for item in data]
    return data
