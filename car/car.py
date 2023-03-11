import pygame
import time
import math
from utils import scale_image, blit_rotate_center
import numpy as np
import random

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

ORDER = [
    (173, 118),
    (62, 188),
    (61, 371),
    (134, 554),
    (290, 712),
    (407, 555),
    (590, 528),
    (617, 710),
    (739, 555),
    (683, 366),
    (426, 354),
    (570, 261),
    (730, 101),
    (501, 76),
    (283, 131),
    (231, 411)
]

ACTIONS =  {
    "UP" : 0,
    "RIGHT" : 1,
    "DOWN" : 2,
    "LEFT" : 3,
    "NOOP" : 4

}
REWARD_UNIT = 10


FIRST_GOAL_CENTROID = (179, 256)
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

GOALS = scale_image(pygame.image.load("imgs/goals.png"), 0.9)
GOALS_MASK = pygame.mask.from_surface(GOALS)
GOALS_MASK_COMPONENTS = GOALS_MASK.connected_components()

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

beam_surface = pygame.Surface((WIDTH, HEIGHT))

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")


FPS = 10


def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)
    pygame.display.update()

def get_mask_components_collided(car):
    for component in GOALS_MASK_COMPONENTS:
        if car.collide(component) != None:
            return component
    return None

def get_goal_mask(centroid):
    for component in GOALS_MASK_COMPONENTS:
        if component.centroid() == centroid:
            return component
    return None
def order_goals_mask():
    ret = []
    for el in ORDER:
        ret.append(get_goal_mask(el))
    return ret

GOALS_MASK_COMPONENTS = order_goals_mask()



class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = 8
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.radars = []

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        self.radars = []
        for radar_angle in (-180,-135,-90,-45, 0, 45, 90,135, 180,225, 270,315, 360):
            self.radar(radar_angle)
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.vel = 0
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0

    def radar(self, radar_angle):
        length = 0
        x_car = self.x+10
        y_car = self.y+25
        x = int(x_car)
        y = int(y_car)
        try:
            while not WIN.get_at((x, y)) == pygame.Color(0, 0, 0, 255) and length < 200:
                length += 1
                x = int(self.x +
                        math.cos(math.radians(self.angle + radar_angle)) * length)
                y = int(self.y -
                        math.sin(math.radians(self.angle + radar_angle)) * length)
        except IndexError:
            pass
        
        if True:
            pygame.draw.line(WIN, (225, 225, 225, 225), (x_car,y_car),
                            (x, y), 1)
            pygame.draw.circle(WIN, (0, 225, 0, 0), (x, y), 3)

        dist = int(
            math.sqrt(
                math.pow(x_car - x, 2) +
                math.pow(y_car - y, 2)))

        self.radars.append([radar_angle, dist])
        


        

class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel*0.2
        self.move()


def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward()
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backward()
    if not moved:
        player_car.reduce_speed()



class CarEnv():
    def __init__(self,player_car=None):
        
        self.player_car = PlayerCar(max_vel = 8, rotation_vel = 8) if player_car == None else player_car
        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
          (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0)),
            (GOALS, (0, 0))]
        self.clock = pygame.time.Clock()
        self.player_car.draw(WIN)
        self.reset()

    def get_observation_space_size(self):
        return 19
    def get_action_space_size(self):
        return len(ACTIONS)
    def sample_from_action_space(self):
        return random.randint(0,len(ACTIONS)-1)

    def reset(self):
        self.n_goals_reached = 0
        self.n_iter = 0
        self.player_car.reset()
        self.run = True
        self.last_goal_reached =  len(ORDER)-1
        return self.get_state()
    
    def get_state(self):
        next_goal = (self.last_goal_reached+1)%len(GOALS_MASK_COMPONENTS)
        next_goal_centroid = GOALS_MASK_COMPONENTS[next_goal].centroid()
        speed = self.player_car.vel
        angle = self.player_car.angle
        x = self.player_car.x
        y = self.player_car.y

        return np.array([
            x,
            y,
            next_goal_centroid[0],
            next_goal_centroid[1],
            speed,
            angle,
            *[t[1] for t in self.player_car.radars],
        ],dtype=np.float32)


    def player_step(self):
        keys = pygame.key.get_pressed()
        moved = False

        if keys[pygame.K_a]:
            self.step(ACTIONS["LEFT"])
        if keys[pygame.K_d]:
            self.step(ACTIONS["RIGHT"])
        if keys[pygame.K_w]:
            moved = True
            self.step(ACTIONS["UP"])
        if keys[pygame.K_s]:
            moved = True
            self.step(ACTIONS["DOWN"])
        if not moved:
            self.step(ACTIONS["NOOP"])
            
         
    def step(self,action):
        self.n_iter += 1
        if self.run == False:
            raise Exception("Game is over")
        reward = 0
        self.clock.tick(FPS)
        draw(WIN, self.images, self.player_car)

        old_x = self.player_car.x
        old_y = self.player_car.y
        
        T = 1
        if action == ACTIONS["LEFT"]:
            self.player_car.rotate(left=True)
        if action == ACTIONS["RIGHT"]:
            self.player_car.rotate(right=True)
        if action == ACTIONS["UP"]:
            self.player_car.move_forward()
 
        if action == ACTIONS["DOWN"]:
            self.player_car.move_backward()
   
        if  action== ACTIONS["NOOP"]:
            self.player_car.reduce_speed()


        self.player_car.draw(WIN)

        next_goal = (self.last_goal_reached+1)%len(GOALS_MASK_COMPONENTS)
        next_goal_centroid = GOALS_MASK_COMPONENTS[next_goal].centroid()

        if abs(self.player_car.x-next_goal_centroid[0])+abs(self.player_car.y-next_goal_centroid[1])<abs(old_x-next_goal_centroid[0])+abs(old_y-next_goal_centroid[1]):
            reward = REWARD_UNIT/10
        else:
            reward = -REWARD_UNIT/10
        
        goal_collided = get_mask_components_collided(self.player_car)

        if goal_collided is not None and goal_collided.centroid() != ORDER[self.last_goal_reached]:
            reward = REWARD_UNIT
            self.last_goal_reached = (self.last_goal_reached+1)%len(GOALS_MASK_COMPONENTS)
            self.n_goals_reached += 1

        if self.player_car.collide(TRACK_BORDER_MASK) != None or self.n_iter > 1_000 * (self.n_goals_reached+1):
            reward = -REWARD_UNIT
            self.run = False
        
        return self.get_state(), reward, not self.run