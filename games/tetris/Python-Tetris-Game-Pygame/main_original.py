import pygame
import sys
from game import Game
from colors import Colors

import os
import json

pygame.init()

title_font = pygame.font.Font(None, 40)
score_surface = title_font.render("Score", True, Colors.white)
next_surface = title_font.render("Next", True, Colors.white)
game_over_surface = title_font.render("GAME OVER", True, Colors.white)

score_rect = pygame.Rect(320, 55, 170, 60)
next_rect = pygame.Rect(320, 215, 170, 180)

screen = pygame.display.set_mode((500, 620))
pygame.display.set_caption("Python Tetris")

clock = pygame.time.Clock()

game = Game()

GAME_UPDATE = pygame.USEREVENT
pygame.time.set_timer(GAME_UPDATE, 1200) # 2800 for vision-text
CACHE_DIR = f"cache/tetris"

# save state
os.makedirs(CACHE_DIR, exist_ok=True)
state_json_path = os.path.join(CACHE_DIR, "state.json")
state_path = os.path.join(CACHE_DIR, "screenshot.png")

def dump_game_state(game, file_path):
    state = {
        "score": game.score,
        "grid": {
            "num_rows": game.grid.num_rows,
            "num_cols": game.grid.num_cols,
            "cells": game.grid.grid
        },
        "current_block": {
            "id": game.current_block.id,
            "positions": [
                {"row": pos.row, "column": pos.column}
                for pos in game.current_block.get_cell_positions()
            ]
        }
    }
    
    with open(file_path, "w") as outfile:
        json.dump(state, outfile, indent=4)

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT or event.type == pygame.KEYDOWN or (event.type == GAME_UPDATE and game.game_over == False):
			print("saving state to...")
			print(state_path)
			print(state_json_path)
			pygame.image.save(screen, state_path)
			dump_game_state(game, state_json_path)

		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		
		if event.type == pygame.KEYDOWN:
			if game.game_over == True:
				game.game_over = False
				game.reset()
			if event.key == pygame.K_LEFT and game.game_over == False:
				game.move_left()
			if event.key == pygame.K_RIGHT and game.game_over == False:
				game.move_right()
			if event.key == pygame.K_DOWN and game.game_over == False:
				game.move_down()
				game.update_score(0, 1)
			if event.key == pygame.K_UP and game.game_over == False:
				game.rotate()
		if event.type == GAME_UPDATE and game.game_over == False:
			game.move_down()

	#Drawing
	score_value_surface = title_font.render(str(game.score), True, Colors.white)

	screen.fill(Colors.dark_blue)
	screen.blit(score_surface, (365, 20, 50, 50))
	screen.blit(next_surface, (375, 180, 50, 50))

	if game.game_over == True:
		screen.blit(game_over_surface, (320, 450, 50, 50))

	pygame.draw.rect(screen, Colors.light_blue, score_rect, 0, 10)
	screen.blit(score_value_surface, score_value_surface.get_rect(centerx = score_rect.centerx, 
	centery = score_rect.centery))
	pygame.draw.rect(screen, Colors.light_blue, next_rect, 0, 10)
	game.draw(screen)

	pygame.display.update()

	clock.tick(60)