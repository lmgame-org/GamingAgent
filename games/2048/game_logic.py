import json
import sys
import argparse
import pygame
import os
from game import playGame

# Default window size
DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500

# Load JSON data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
constants_path = os.path.join(BASE_DIR, "constants.json")
c = json.load(open(constants_path, "r"))

# Argument parser for width and height
parser = argparse.ArgumentParser(description="Run 2048 Game with custom window size")
parser.add_argument("-wd", "--width", type=int, default=DEFAULT_WIDTH, help="Set window width")
parser.add_argument("-ht", "--height", type=int, default=DEFAULT_HEIGHT, help="Set window height")  # Changed -h to -ht
args = parser.parse_args()

# Set window size
size = (args.width, args.height)

# Set up pygame
pygame.init()
size = (args.width, args.height)
screen = pygame.display.set_mode(size)

# Set font according to JSON data specifications
my_font = pygame.font.SysFont(c["font"], c["font_size"], bold=True)

if __name__ == "__main__":
    playGame("light", 2048, size)
