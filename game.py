"""
Python clone of 2048 found on play2048.co using Pygame.

Use stable-baselines3 to train an AI to play 2048.
"""

import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from gym import spaces
from pygame.locals import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Constants
WIDTH = 450
HEIGHT = 450
FPS = 60
TILE_SIZE = 100
AI_MODE = True

# Calculate padding
TILE_PADDING = (WIDTH - (TILE_SIZE * 4)) / 5

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


class Tile(pygame.sprite.Sprite):
    """
    Tile class for each tile on the board.
    """

    def __init__(self, value):
        """
        Initialize a Tile object.
        """
        super().__init__()
        self.value = value
        self.image = self.get_image()

    def get_image(self):
        """
        Get the image of the tile.
        """
        image = pygame.image.load(f"images/{self.value}.png")
        image = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
        return image

    def update(self):
        """
        Update the tile.
        """
        self.image = self.get_image()

    def draw(self, surface, x, y):
        """
        Draw the tile
        """
        surface.blit(self.image, (x, y))


class Board:
    """
    Board class that holds a 4x4 grid of Tile(s) and logic for moving/merging
    """

    def __init__(self):
        """
        Initialize a Board object.
        """
        self.grid = [[Tile(0) for y in range(4)] for x in range(4)]
        self.score = 0
        self.moved = False

        # Add two tiles to the board
        self.add_tile(2)

    def add_tile(self, amount):
        """
        Add (amount) tiles to empty positions (Tile.value = 0)
        """
        for _ in range(amount):
            x = random.randint(0, 3)
            y = random.randint(0, 3)
            while self.grid[x][y].value != 0:
                x = random.randint(0, 3)
                y = random.randint(0, 3)
            self.grid[x][y].value = 2

    def valid_move(self, direction):
        """
        Check if the given move is valid.
        """
        if direction == "down":
            for x in range(4):
                for y in range(3, 0, -1):
                    if self.grid[x][y].value == 0:
                        return True
                    elif self.grid[x][y].value == self.grid[x][y - 1].value:
                        return True
        elif direction == "up":
            for x in range(4):
                for y in range(3):
                    if self.grid[x][y].value == 0:
                        return True
                    elif self.grid[x][y].value == self.grid[x][y + 1].value:
                        return True
        elif direction == "left":
            for x in range(3):
                for y in range(4):
                    if self.grid[x][y].value == 0:
                        return True
                    elif self.grid[x][y].value == self.grid[x + 1][y].value:
                        return True
        elif direction == "right":
            for x in range(3, 0, -1):
                for y in range(4):
                    if self.grid[x][y].value == 0:
                        return True
                    elif self.grid[x][y].value == self.grid[x - 1][y].value:
                        return True
        return False

    def move_tiles(self, direction):
        """
        Move tiles in the given direction.
        """
        if direction == "down":
            self.slide_down()
        elif direction == "up":
            self.slide_up()
        elif direction == "left":
            self.slide_left()
        elif direction == "right":
            self.slide_right()

        # Add a new tile to the board
        if self.moved:
            self.add_tile(1)
            self.moved = False

    def slide_down(self, recursion_step=3):
        """
        Slide all tiles down.
        """
        if recursion_step == 0:
            return

        for x in range(4):
            for y in range(3, 0, -1):
                if self.grid[x][y].value == 0:
                    self.grid[x][y].value = self.grid[x][y - 1].value
                    self.grid[x][y - 1].value = 0
                    self.moved = True
                elif self.grid[x][y].value == self.grid[x][y - 1].value:
                    self.grid[x][y].value *= 2
                    self.grid[x][y - 1].value = 0
                    self.score += self.grid[x][y].value
                    self.moved = True
        self.slide_down(recursion_step - 1)

    def slide_up(self, recursion_step=3):
        """
        Slide all tiles up.
        """
        if recursion_step == 0:
            return

        for x in range(4):
            for y in range(3):
                if self.grid[x][y].value == 0:
                    self.grid[x][y].value = self.grid[x][y + 1].value
                    self.grid[x][y + 1].value = 0
                    self.grid[x]
                    self.moved = True
                elif self.grid[x][y].value == self.grid[x][y + 1].value:
                    self.grid[x][y].value *= 2
                    self.grid[x][y + 1].value = 0
                    self.score += self.grid[x][y].value
                    self.moved = True
        self.slide_up(recursion_step - 1)

    def slide_left(self, recursion_step=3):
        """
        Slide all tiles left.
        """
        if recursion_step == 0:
            return

        for y in range(4):
            for x in range(3):
                if self.grid[x][y].value == 0:
                    self.grid[x][y].value = self.grid[x + 1][y].value
                    self.grid[x + 1][y].value = 0
                    self.moved = True
                elif self.grid[x][y].value == self.grid[x + 1][y].value:
                    self.grid[x][y].value *= 2
                    self.grid[x + 1][y].value = 0
                    self.score += self.grid[x][y].value
                    self.moved = True
        self.slide_left(recursion_step - 1)

    def slide_right(self, recursion_step=3):
        """
        Slide all tiles right.
        """
        if recursion_step == 0:
            return

        for y in range(4):
            for x in range(3, 0, -1):
                if self.grid[x][y].value == 0:
                    self.grid[x][y].value = self.grid[x - 1][y].value
                    self.grid[x - 1][y].value = 0
                    self.moved = True
                elif self.grid[x][y].value == self.grid[x - 1][y].value:
                    self.grid[x][y].value *= 2
                    self.grid[x - 1][y].value = 0
                    self.score += self.grid[x][y].value
                    self.moved = True
        self.slide_right(recursion_step - 1)

    def draw(self, screen):
        """
        Draw our grid:
            - 4x4 grid with TILE_PADDING between the tiles of TILE_SIZE
        """
        for x in range(4):
            for y in range(4):
                self.grid[x][y].update()
                pos_x = (x * (TILE_SIZE + TILE_PADDING)) + TILE_PADDING
                pos_y = (y * (TILE_SIZE + TILE_PADDING)) + TILE_PADDING
                self.grid[x][y].draw(screen, pos_x, pos_y)

    def get_state(self):
        """
        Get the state of the board
        """
        state = []
        for x in range(4):
            for y in range(4):
                state.append(self.grid[x][y].value)
        return np.array(state)

    def is_game_over(self):
        """
        Check if the game is over.
        """
        for x in range(4):
            for y in range(4):
                if self.grid[x][y].value == 0:
                    return False
                elif x < 3 and self.grid[x][y].value == self.grid[x + 1][y].value:
                    return False
                elif y < 3 and self.grid[x][y].value == self.grid[x][y + 1].value:
                    return False
        return True

    def get_empty_tiles(self):
        """
        Get the empty tiles
        """
        empty_tiles = []
        for x in range(4):
            for y in range(4):
                if self.grid[x][y].value == 0:
                    empty_tiles.append((x, y))
        return empty_tiles

    def calculate_reward_value(self):
        """
        Calculate the reward value

        Boards with highest number in a corner with a monotonic row are optimal
        """
        reward = 0
        highest_tile = 0
        for x in range(4):
            for y in range(4):
                if self.grid[x][y].value > highest_tile:
                    highest_tile = self.grid[x][y].value
        reward += highest_tile

        # Check if highest tile is in a corner
        if self.grid[0][0].value == highest_tile:
            reward += 10
        elif self.grid[3][3].value == highest_tile:
            reward += 10
        elif self.grid[0][3].value == highest_tile:
            reward += 10
        elif self.grid[3][0].value == highest_tile:
            reward += 10

        # Check if rows are monotonic
        for x in range(4):
            for y in range(4):
                if y < 3 and self.grid[x][y].value >= self.grid[x][y + 1].value:
                    reward += 5
                if x < 3 and self.grid[x][y].value >= self.grid[x + 1][y].value:
                    reward += 5
        return reward


class Game:
    """
    Game class that holds the board and handles user input
    """

    def __init__(self):
        """
        Initialize the game
        """
        self.board = Board()

    def play(self):
        """
        Play the game
        """
        # Set up the screen
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2048")
        clock = pygame.time.Clock()

        # Game loop
        while True:
            # Update title
            pygame.display.set_caption("2048 - Score: {}".format(self.board.score))

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.board.move_tiles("up")
                    elif event.key == pygame.K_DOWN:
                        self.board.move_tiles("down")
                    elif event.key == pygame.K_LEFT:
                        self.board.move_tiles("left")
                    elif event.key == pygame.K_RIGHT:
                        self.board.move_tiles("right")

            # Draw the screen
            screen.fill(BLACK)
            self.board.draw(screen)
            pygame.display.flip()

            # Limit to 60 frames per second
            clock.tick(60)


class Game2048Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_timesteps=1000):
        self.game = Game()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2048, shape=(16,), dtype=np.int32
        )
        self.max_timesteps = max_timesteps
        self.timestep = 0

    def step(self, action):
        if self.max_timesteps > 0:
            self.timestep += 1
            if self.timestep >= self.max_timesteps:
                return self.game.board.get_state(), 0, True, {}

        self.game.board.move_tiles(["down", "up", "left", "right"][action])
        reward = self.game.board.calculate_reward_value()
        done = self.game.board.is_game_over()
        if done:
            reward = -100
        info = {}
        return self.game.board.get_state(), float(reward), done, info

    def reset(self):
        self.game = Game()
        self.timestep = 0
        return self.game.board.get_state()

    def render(self, mode="human", close=False):
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2048 - Score: {}".format(self.game.board.score))
        clock = pygame.time.Clock()
        screen.fill(BLACK)
        self.game.board.draw(screen)
        pygame.display.flip()
        clock.tick(60)


def main():
    env = Game2048Env()
    check_env(env)

    if AI_MODE:
        train = False
        if train:
            # Train the model
            env = Game2048Env()
            model = PPO(
                "MlpPolicy", env, verbose=1, tensorboard_log="./2048_tensorboard/"
            )
            model.learn(total_timesteps=1_000_000)
            model.save("2048")
        else:
            # Load the model
            model = PPO.load("2048")

            # Play the game X times, awaiting user input to continue after each game
            env = Game2048Env(max_timesteps=0)
            obs = env.reset()
            scores = []
            for i in range(5):
                done = False
                while not done:
                    action, _states = model.predict(obs)
                    obs, reward, done, info = env.step(action)
                    env.render()
                scores.append(env.game.board.score)
                print("Game {} - Score: {}".format(i + 1, env.game.board.score))
                env.render()
                input("Press enter to continue...")
                env.reset()

            print(f"Average score: {sum(scores) / len(scores)}")
            for score in scores:
                print(score)

    else:
        game = Game()
        game.play()


if __name__ == "__main__":
    main()
