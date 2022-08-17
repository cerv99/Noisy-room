import numpy as np
import cv2
from PIL import Image
import pickle
from random import choice


# in this environment, the vision of hunters will be limited and they can communicate
def judge(player, food):
    if player.x == food.x and abs(food.y - player.y) == 1:
        return True
    if player.y == food.y and abs(player.x - food.x) == 1:
        return True
    return False


def vision(player, food, vis_range=2):
    if abs(player.x - food.x) <= vis_range and abs(player.y - food.y) <= vis_range:
        return True
    return False


class Cube:
    def __init__(self, size, name='player1'):  # Initialize the position of hunter and prey
        self.size = size
        self.name = name
        # self.x = np.random.randint(0, self.size-1)
        # self.y = np.random.randint(0, self.size-1)
        if name == 'player1':
            self.x = 5
            self.y = 5
        elif name == 'player2':
            self.x = 2
            self.y = 2
        if name == 'food':
            self.x = 7
            self.y = 7

    def __str__(self):
        return f'{self.x},{self.y}'

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choise):
        if choise == 0:
            self.move(x=0, y=1)
        elif choise == 1:
            self.move(x=0, y=-1)
        elif choise == 2:
            self.move(x=1, y=0)
        elif choise == 3:
            self.move(x=-1, y=0)
        elif choise == 4:
            self.move(x=0, y=0)

    def move(self, x=False, y=False, ran=False):
        if ran:
            choise = choice([0, 1, 2, 3, 4])
            if choise == 0:
                x = 0
                y = 1
            elif choise == 1:
                x = 0
                y = -1
            elif choise == 2:
                x = 1
                y = 0
            elif choise == 3:
                x = -1
                y = 0
            elif choise == 4:
                x = 0
                y = 0

        if not x:
            self.x += 0
            # self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += 0
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        if self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        if self.y > self.size - 1:
            self.y = self.size - 1


class envCube:
    SIZE = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_VALUES = 5
    RETURN_IMAGE = False

    FOOD_REWARD = 25
    MOVE_PENALITY = -1

    # Set colors
    d = {1: (255, 0, 0),  # blue
         2: (0, 255, 0)}  # green
    PLAYER_N = 1
    FOOD_N = 2

    # reset environment
    def reset(self):
        self.player1 = Cube(self.SIZE, 'player1')
        self.player2 = Cube(self.SIZE, 'player2')
        self.food = Cube(self.SIZE, 'food')
        VIS_RANGE = 2
        if self.RETURN_IMAGE:
            observation = np.array(self.get_image())
        else:
            if vision(self.player1, self.food, vis_range=VIS_RANGE):
                if vision(self.player2, self.food, vis_range=VIS_RANGE) or vision(self.player2, self.player1,
                                                                                  vis_range=VIS_RANGE):
                    observation = (self.player1 - self.food) + (self.player2 - self.food)
                else:
                    observation = (self.player1 - self.food) + (-9, -9)
            elif vision(self.player2, self.food, vis_range=VIS_RANGE):
                if vision(self.player2, self.player1,
                          vis_range=VIS_RANGE):
                    observation = (self.player1 - self.food) + (self.player2 - self.food)
                else:
                    observation = (-9, -9) + (self.player2 - self.food)
            else:
                observation = (-9, -9, -9, -9)

        self.episode_step = 0

        return observation

    def step(self, action):
        self.episode_step += 1
        self.player1.action(action)
        self.player2.action(action)
        self.food.move(ran=True)
        VIS_RANGE = 2
        if self.RETURN_IMAGE:
            new_observation = np.array(self.get_image())
        else:
            if vision(self.player1, self.food, vis_range=VIS_RANGE):
                if vision(self.player2, self.food, vis_range=VIS_RANGE) or vision(self.player2, self.player1,
                                                                                  vis_range=VIS_RANGE):
                    new_observation = (self.player1 - self.food) + (self.player2 - self.food)
                else:
                    new_observation = (self.player1 - self.food) + (-9, -9)
            elif vision(self.player2, self.food, vis_range=VIS_RANGE):
                if vision(self.player2, self.player1,
                          vis_range=VIS_RANGE):
                    new_observation = (self.player1 - self.food) + (self.player2 - self.food)
                else:
                    new_observation = (-9, -9) + (self.player2 - self.food)
            else:
                new_observation = (-9, -9, -9, -9)

        if judge(self.player1, self.food) and judge(self.player2, self.food):
            reward = self.FOOD_REWARD
        else:
            reward = self.MOVE_PENALITY

        done = False

        if judge(self.player1, self.food) and judge(self.player2, self.food) or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((320, 320))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        env[self.player1.x][self.player1.y] = self.d[self.PLAYER_N]
        env[self.player2.x][self.player2.y] = self.d[self.PLAYER_N]
        img = Image.fromarray(env, 'RGB')
        return img

    def get_qtable(self, q_table_name=None):
        # Initialize Q-table
        if q_table_name is None:
            q_table = {}
            for x1 in range(-self.SIZE + 1, self.SIZE):
                for y1 in range(-self.SIZE + 1, self.SIZE):
                    for x2 in range(-self.SIZE + 1, self.SIZE):
                        for y2 in range(-self.SIZE + 1, self.SIZE):
                            q_table[(x1, y1, x2, y2)] = [np.random.randint(-5, 0) for i in
                                                         range(self.ACTION_SPACE_VALUES)]
        else:
            with open(q_table_name, 'rb') as f:
                q_table = pickle.load(f)
        return q_table
