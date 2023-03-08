import pygame
import time
import random
import numpy as np



#yellow = (255, 255, 102)
#red = (213, 50, 80)

#blue = (50, 153, 213)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
 
DIS_WIDTH = 200
DIS_HEIGHT = 200

SNAKE_BLOCK = 10

SNAKE_SPEED = 100

REWARD_UNIT = 10.0

ACTIONS = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3
}

DIRECTIONS = {
    'STRAIGHT': 0,
    'RIGHT': 1,
    'LEFT': 2
}


class SnakeGame():
    def __init__(self):
        pygame.init()
        self.dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
        self.clock = pygame.time.Clock()
        self.reset()

    def get_observation_space_size(self):
        return self.get_state().shape[0]
    
    def get_action_space_size(self):
        return len(DIRECTIONS)
    
    def sample_from_action_space(self):
        return random.randint(0, len(DIRECTIONS)-1)
        

    def draw_snake(self):
        for x in self.snake_list:
            pygame.draw.rect(self.dis, BLACK, [x[0], x[1], SNAKE_BLOCK, SNAKE_BLOCK])
    
    def reset(self):

        self.direction = ACTIONS['RIGHT']
        self.frame_iteration = 0
        self.game_close = False

        self.x1 = DIS_WIDTH / 2
        self.y1 = DIS_HEIGHT / 2

        self.x1_change = 0
        self.y1_change = 0

        self.snake_list = []
        self.length_of_snake = 1

        self.foodx = round(random.randrange(0, DIS_WIDTH - SNAKE_BLOCK) / 10.0) * 10.0
        self.foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0

        return self.get_state()


    def is_collision(self,point):
        # wall and snake collision check
        return point[0] >= DIS_WIDTH or point[0] < 0 or \
                point[1] >= DIS_HEIGHT or point[1] < 0 or \
                point in self.snake_list[:-1]

    
    def get_state(self):
        point_sx = [self.x1-SNAKE_BLOCK,self.y1]
        point_dx = [self.x1+SNAKE_BLOCK,self.y1]
        point_up = [self.x1,self.y1-SNAKE_BLOCK]
        point_dw = [self.x1,self.y1+SNAKE_BLOCK]
        
        dir_up = 1 if self.direction == ACTIONS['UP'] else 0
        dir_dw = 1 if self.direction == ACTIONS['DOWN'] else 0
        dir_sx = 1 if self.direction == ACTIONS['LEFT'] else 0
        dir_dx = 1 if self.direction == ACTIONS['RIGHT'] else 0
        
        return np.array(
            [
                self.x1,
                self.y1,

                self.foodx<self.x1,
                self.foodx>self.x1,
                self.foody>self.y1,
                self.foody<self.y1,

                dir_up,
                dir_dw,
                dir_sx,
                dir_dx,

                # straight
                (dir_dx and self.is_collision(point_dx) or
                 dir_up and self.is_collision(point_up) or
                 dir_dw and self.is_collision(point_dw) or
                 dir_sx and self.is_collision(point_sx)),

                # right
                (dir_up and self.is_collision(point_dx) or
                 dir_dw and self.is_collision(point_sx) or
                 dir_sx and self.is_collision(point_up) or
                 dir_dx and self.is_collision(point_dw)),

                # left
                (dir_up and self.is_collision(point_sx) or
                 dir_dw and self.is_collision(point_dx) or
                 dir_sx and self.is_collision(point_dw) or
                 dir_dx and self.is_collision(point_up)),
                

            ], 
                dtype=int
        )
        

    def step(self,action_dir):
        self.frame_iteration += 1
        reward = 0
        if action_dir == DIRECTIONS['STRAIGHT']:
            action = self.direction
        elif action_dir == DIRECTIONS['RIGHT']: 
            action = (self.direction+1)%4
        elif action_dir == DIRECTIONS['LEFT']:
            action = (self.direction-1)%4

        self.direction = action

        old_x = self.x1
        old_y = self.y1

        if self.game_close:
            raise Exception("Game is over")
        if action == ACTIONS['LEFT']:
            self.x1_change = -SNAKE_BLOCK
            self.y1_change = 0
        elif action == ACTIONS['RIGHT']:
            self.x1_change = SNAKE_BLOCK
            self.y1_change = 0
        elif action == ACTIONS['UP']:
            self.y1_change = -SNAKE_BLOCK
            self.x1_change = 0
        elif action == ACTIONS['DOWN']:
            self.y1_change = SNAKE_BLOCK
            self.x1_change = 0

        
        if self.x1 >= DIS_WIDTH or self.x1 < 0 or self.y1 >= DIS_HEIGHT or self.y1 < 0:
            self.game_close = True
        self.x1 += self.x1_change
        self.y1 += self.y1_change

        if abs(self.x1 - self.foodx) + abs(self.y1-self.foody) < abs(old_x-self.foodx)+abs(old_y-self.foody):
            reward = REWARD_UNIT/10
        else:
            reward = -REWARD_UNIT/10

        self.dis.fill(WHITE)

        pygame.draw.rect(self.dis, GREEN, [self.foodx, self.foody, SNAKE_BLOCK, SNAKE_BLOCK])

        snake_Head = []
        snake_Head.append(self.x1)
        snake_Head.append(self.y1)
        self.snake_list.append(snake_Head)
        if len(self.snake_list) > self.length_of_snake:
            del self.snake_list[0]  
        for x in self.snake_list[:-1]:
            if x == snake_Head or self.frame_iteration > 100 * len(self.snake_list)**2:
                self.game_close = True
        
        self.draw_snake()

        pygame.display.update()

        if self.x1 == self.foodx and self.y1 == self.foody:
            reward = REWARD_UNIT
            self.foodx = round(random.randrange(0, DIS_WIDTH - SNAKE_BLOCK) / 10.0) * 10.0
            self.foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0
            self.length_of_snake += 1

        self.clock.tick(SNAKE_SPEED)

        if self.game_close:
            reward = -REWARD_UNIT
        infos = {
            'score': self.length_of_snake,
        }
        return self.get_state(), reward, self.game_close, infos
    
    
        
