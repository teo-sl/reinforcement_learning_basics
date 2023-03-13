import torch
import torch.nn as nn
from car import ACTIONS


class Network(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,len(ACTIONS) ),
        )

    def forward(self, x):
        return self.net(x)
    def act(self,obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self.forward(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action