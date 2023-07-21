from car import CarEnv,WIN,FPS,draw,ACTIONS
from net_model import Network
import torch
import pygame
import random

CHECKPOINT_PATH = "./car_target_net_50k.pth"


env = CarEnv()
net = Network(env)
net.load_state_dict(torch.load(CHECKPOINT_PATH))
run = True


done = False
env = CarEnv()
obs = env.reset()
while not done:
    #if random.random() < 0.05:
    #    action = random.randint(0,len(ACTIONS)-1)
    #else:                           
    action = net.act(obs)
    obs, _, done = env.step(action)
    
pygame.quit()