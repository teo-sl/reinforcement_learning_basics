import torch
from torch import nn

class Network(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.get_observation_space_size(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, env.get_action_space_size()),
        )

    def forward(self, x):
        return self.net(x)
    def act(self,obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self.forward(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action