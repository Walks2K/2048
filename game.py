"""
Python clone of 2048 found on play2048.co using Pygame.
"""

import pygame
import random
import numpy as np

# Constants
WIDTH = 450
HEIGHT = 450
FPS = 60
TILE_SIZE = 100

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
            pygame.display.set_caption("2048 - Score: {}".format(
                self.board.score))

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


if __name__ == "__main__":
    game = Game()
    game.play()