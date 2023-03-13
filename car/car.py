import pygame
import time
import math
from utils import scale_image, blit_rotate_center
import numpy as np
import random

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

ORDER = [(176, 144),
 (161, 93),
 (115, 74),
 (69, 101),
 (61, 147),
 (60, 216),
 (61, 302),
 (60, 400),
 (64, 477),
 (109, 529),
 (196, 618),
 (303, 722),
 (376, 724),
 (409, 665),
 (406, 576),
 (425, 513),
 (478, 481),
 (547, 486),
 (596, 539),
 (601, 613),
 (608, 697),
 (644, 727),
 (691, 729),
 (732, 696),
 (738, 612),
 (744, 452),
 (737, 404),
 (708, 372),
 (670, 364),
 (545, 365),
 (434, 360),
 (402, 316),
 (435, 265),
 (514, 262),
 (659, 260),
 (726, 241),
 (741, 178),
 (737, 118),
 (704, 79),
 (646, 76),
 (535, 77),
 (411, 75),
 (339, 75),
 (295, 93),
 (278, 145),
 (280, 228),
 (281, 330),
 (271, 389),
 (227, 411),
 (177, 374),
 (178, 285)]

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

#FINISH = pygame.image.load("imgs/finish.png")
#FINISH_MASK = pygame.mask.from_surface(FINISH)
#FINISH_POSITION = (130, 250)

#RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
#GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

beam_surface = pygame.Surface((WIDTH, HEIGHT))

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")


FPS = 60


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
        #self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = 4
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
        for angle in [0,22.5,45,67.5,80,90,100,112.5,135,157.5,180]:
            radar_angle = angle 
            self.radar(radar_angle)
        blit_rotate_center(win, (self.x, self.y), self.angle)
        

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
        circle = pygame.draw.circle(WIN, (255, 0, 0,0), (self.x, self.y), 3)
        surf = WIN.subsurface(circle)
        car_mask = pygame.mask.from_surface(surf)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0

    def radar(self, radar_angle):
        length = 0
        x_car = self.x
        y_car = self.y
        x = int(x_car)
        y = int(y_car)
        try:
            while not WIN.get_at((x, y)) == pygame.Color(0, 0, 0, 255) and length < 50:
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
            if radar_angle == 90:
                color = (255, 0, 0, 0)
            else:
                color = (0, 255,0, 0)
            pygame.draw.circle(WIN, color, (x, y), 3)

        dist = int(
            math.sqrt(
                math.pow(x_car - x, 2) +
                math.pow(y_car - y, 2)))

        self.radars.append([radar_angle, dist])
        


        

class PlayerCar(AbstractCar):
    #IMG = RED_CAR
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
           (TRACK_BORDER, (0, 0)),
            (GOALS, (0, 0))]
        self.clock = pygame.time.Clock()
        self.player_car.draw(WIN)
        self.reset()

    def get_observation_space_size(self):
        return 21
    def get_action_space_size(self):
        return len(ACTIONS)
    def sample_from_action_space(self):
        return random.randint(0,len(ACTIONS)-1)

    def reset(self):
        self.n_no_consecutive_move = 0
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

        phi = math.atan2(x-next_goal_centroid[0],y-next_goal_centroid[1])
        phi = math.degrees(phi)
        angle = (angle+360)%360
        phi = (phi+360)%360

        diff = (angle-phi+540)%360-180

        if -5 <= diff <= 5:
            go_left = 0
            go_right = 0
            go_straight = 1
        elif diff < -5:
            go_left = 1
            go_right = 0
            go_straight = 0
        elif diff > 5:
            go_left = 0
            go_right = 1
            go_straight = 0

        dist_x = next_goal_centroid[0]-x
        dist_y = next_goal_centroid[1]-y

        if dist_x > 0:
            is_left = 1
            is_right = 0
        else:
            is_left = 0
            is_right = 1
        if dist_y > 0:
            is_up = 0
            is_down = 1
        else:
            is_up = 1
            is_down = 0

        return np.array([
            self.n_no_consecutive_move,
            is_left,
            is_right,
            is_up,
            is_down,
            speed/8.0,
            angle/360.0,
            go_left,
            go_straight,
            go_right,
            *[t[1]/50.0 for t in self.player_car.radars],
        ],dtype=np.float32)


    def player_step(self):
        keys = pygame.key.get_pressed()
        moved = False

        if keys[pygame.K_a]:
            return self.step(ACTIONS["LEFT"])
        if keys[pygame.K_d]:
            return self.step(ACTIONS["RIGHT"])
        if keys[pygame.K_w]:
            moved = True
            return self.step(ACTIONS["UP"])
        if keys[pygame.K_s]:
            moved = True
            return self.step(ACTIONS["DOWN"])
        if not moved:
            return self.step(ACTIONS["NOOP"])
            
         
    def step(self,action):
        self.n_iter += 1

        if self.run == False:
            raise Exception("Game is over")
        reward = 0.0
        
        draw(WIN, self.images, self.player_car)

        old_x = self.player_car.x
        old_y = self.player_car.y
        
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
        dist = -(abs(self.player_car.x-next_goal_centroid[0])+abs(self.player_car.y-next_goal_centroid[1]))+(abs(old_x-next_goal_centroid[0])+abs(old_y-next_goal_centroid[1]))
        if dist > 0:
            reward = dist / 8.0
            self.n_no_consecutive_move = 0
        elif self.player_car.x == old_x and self.player_car.y == old_y:
            if self.n_no_consecutive_move > 20:
                reward = -(self.n_no_consecutive_move-20)/10.0
                self.n_no_consecutive_move += 1
            else:
                reward = 0.0
                self.n_no_consecutive_move +=1
        else:
            reward = - 0.1 + dist / 8.0
            self.n_no_consecutive_move = 0

        
        goal_collided = get_mask_components_collided(self.player_car)

        if goal_collided is not None and goal_collided.centroid() != ORDER[self.last_goal_reached]:
            reward = REWARD_UNIT
            self.last_goal_reached = (self.last_goal_reached+1)%len(GOALS_MASK_COMPONENTS)
            reward += self.n_goals_reached * (REWARD_UNIT/15.0)
            self.n_goals_reached += 1

        if self.player_car.collide(TRACK_BORDER_MASK) != None or self.n_iter > 1_000 * (self.n_goals_reached+1):
            reward = -REWARD_UNIT
            self.run = False

        self.clock.tick(FPS)
        
        return self.get_state(), reward, not self.run