from model import Network
from game import SnakeGame
import torch


env = SnakeGame()
net = Network(env)
net.load_state_dict(torch.load('../checkpoints/snake_target_net_.pth'))


obs = env.reset()
done = False
while not done :
    obs, _, done, _ = env.step(net.act(obs))