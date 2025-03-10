import json
import sys
import time
from copy import deepcopy

import pygame
from pygame.locals import *

from logic import *
import os

# TODO: Add a RULES button on start page
# TODO: Add score keeping

CACHE_DIR = "cache/2048"

os.makedirs(CACHE_DIR, exist_ok=True)

# set up pygame for main gameplay
pygame.init()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
constants_path = os.path.join(BASE_DIR, "constants.json")
c = json.load(open(constants_path, "r"))
screen = pygame.display.set_mode(
    (c["size"], c["size"]))
my_font = pygame.font.SysFont(c["font"], c["font_size"], bold=True)
WHITE = (255, 255, 255)


def winCheck(board, status, theme, text_col, size):
    """
    Check game status and display win/lose result.

    Parameters:
        board (list): game board
        status (str): game status
        theme (str): game interface theme
        text_col (tuple): text colour
        size (tuple): (width, height) of the game window
    Returns:
        board (list): updated game board
        status (str): game status
    """
    if status != "PLAY":
        # Create a transparent overlay
        s = pygame.Surface(size, pygame.SRCALPHA)
        s.fill(c["colour"][theme]["over"])
        screen.blit(s, (0, 0))

        # Dynamically adjust font sizes
        title_font_size = max(size[0] // 10, 36)  # Scale with screen width
        subtitle_font_size = max(size[0] // 20, 24)  # Slightly smaller for prompts

        title_font = pygame.font.SysFont(c["font"], title_font_size, bold=True)
        subtitle_font = pygame.font.SysFont(c["font"], subtitle_font_size, bold=True)

        # Display win/lose message
        msg = "YOU WIN!" if status == "WIN" else "GAME OVER!"
        title_text = title_font.render(msg, True, text_col)

        # Calculate centered positions
        title_x = (size[0] - title_text.get_width()) // 2
        title_y = size[1] // 3  # Position at 1/3 height of screen

        # Render restart instructions
        restart_text = subtitle_font.render("Play again? (Y/N)", True, text_col)
        restart_x = (size[0] - restart_text.get_width()) // 2
        restart_y = title_y + title_text.get_height() + 20  # Below title

        # Blit text to screen
        screen.blit(title_text, (title_x, title_y))
        screen.blit(restart_text, (restart_x, restart_y))

        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or \
                        (event.type == pygame.KEYDOWN and event.key == pygame.K_n):
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_y:
                    # 'Y' is pressed to start a new game
                    board = newGame(theme, text_col, size)
                    return board, "PLAY"

    return board, status



def newGame(theme, text_col, size):
    """
    Start a new game by resetting the board.

    Parameters:
        theme (str): game interface theme
        text_col (tuple): text colour
        size (tuple): (width, height) of the game window
    Returns:
        board (list): new game board
    """
    # Clear the board to start a new game
    board = [[0] * 4 for _ in range(4)]
    display(board, theme, size)

    # Dynamically adjust font size based on window size
    font_size = max(size[0] // 15, 24)  # Scales with screen width, min size 24
    title_font = pygame.font.SysFont(c["font"], font_size, bold=True)

    # Render "NEW GAME!" text
    new_game_text = title_font.render("NEW GAME!", True, text_col)

    # Calculate centered position
    text_x = (size[0] - new_game_text.get_width()) // 2
    text_y = (size[1] - new_game_text.get_height()) // 2

    # Blit text to screen at centered position
    screen.blit(new_game_text, (text_x, text_y))
    pygame.display.update()

    # Wait for 1 second before starting game
    time.sleep(1)

    # Fill two random tiles at the beginning
    board = fillTwoOrFour(board, iter=2)
    display(board, theme, size)

    return board


def restart(board, theme, text_col, size):
    """
    Restart the game immediately when 'N' is pressed.
    """
    print("Restarting game...")  # Debugging output
    return newGame(theme, text_col, size)

def display(board, theme, size):
    """
    Display the board 'matrix' on the game window.

    Parameters:
        board (list): game board
        theme (str): game interface theme
        size (tuple): (width, height) of the game window
    """
    screen.fill(tuple(c["colour"][theme]["background"]))
    
    grid_size = 4  # 2048 is a 4x4 grid
    box = size[0] // grid_size  # Adjust tile size dynamically based on window size
    padding = box // 10  # Set padding relative to box size

    # Update font size dynamically based on tile size
    font_size = max(box // 3, 24)  # Ensure font is proportional but not too small
    my_font = pygame.font.SysFont(c["font"], font_size, bold=True)

    for i in range(grid_size):
        for j in range(grid_size):
            colour = tuple(c["colour"][theme][str(board[i][j])])
            pygame.draw.rect(screen, colour, (j * box + padding,
                                              i * box + padding,
                                              box - 2 * padding,
                                              box - 2 * padding), 0)

            if board[i][j] != 0:
                if board[i][j] in (2, 4):
                    text_colour = tuple(c["colour"][theme]["dark"])
                else:
                    text_colour = tuple(c["colour"][theme]["light"])

                # Render number text centered within tile
                text_surface = my_font.render(f"{board[i][j]}", True, text_colour)
                
                # Calculate center position dynamically
                text_x = j * box + (box - text_surface.get_width()) // 2
                text_y = i * box + (box - text_surface.get_height()) // 2
                
                screen.blit(text_surface, (text_x, text_y))
    
    pygame.display.update()


def playGame(theme, difficulty, size):
    """
    Main game loop function.

    Parameters:
        theme (str): game interface theme
        difficulty (int): game difficulty, i.e., max. tile to get
    """
    # Initialise game status
    status = "PLAY"

    # Set text colour according to theme
    text_col = (0, 0, 0) if theme == "light" else (255, 255, 255)

    board = newGame(theme, text_col, size)

    screenshot_path = os.path.abspath(os.path.join(CACHE_DIR, "2048_screenshot.png"))

    pygame.image.save(screen, screenshot_path)

    # Define movement key mappings
    movement_keys = {
        pygame.K_LEFT: "a",
        pygame.K_RIGHT: "d",
        pygame.K_UP: "w",
        pygame.K_DOWN: "s"
    }

    # Main game loop
    while True:
        for event in pygame.event.get():
            # Handle Quit
        
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:


                if event.key == pygame.K_LCTRL or event.key == pygame.K_RCTRL:
                    print("Restarting game...")
                    board = restart(board, theme, text_col, size)
                    continue

                # Handle Movement Keys
                if event.key in movement_keys:
                    key = movement_keys[event.key]
                    print(f"Key mapped: {key}")  # Debugging output

                    # Perform move
                    new_board = move(key, deepcopy(board))

                    # Only update board if there was a change
                    if new_board != board:
                        board = fillTwoOrFour(new_board)
                        display(board, theme, size)

                        # Update game status
                        status = checkGameStatus(board, difficulty)

                        # Check win/lose
                        board, status = winCheck(board, status, theme, text_col, size)
                
                pygame.image.save(screen, screenshot_path)
