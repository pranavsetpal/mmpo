import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(n_obs, n_actions):
        super.__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.RELU(),
            nn.Linear(64, 64),
            nn.RELU(),
            nn.Linear(64, n_actions)
        )
        self.softmax = nn.Softmax()

    def forward(obs):
        action_probs = mlp(obs)
        _, action_idx = torch.max(softmax(action_probs))

        return action_idx
