import pygame
import sys
import argparse
import os
import json
from game import Game
from colors import Colors

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

def main():
    # Parse command-line arguments with argparse.
    parser = argparse.ArgumentParser(
        description="Run Python Tetris in dynamic or frozen mode."
    )
    parser.add_argument("--mode", choices=["dynamic", "frozen"],
                        help="Game update mode: 'dynamic' for continuous updates or 'frozen' for updates on user inactivity.")
    args = parser.parse_args()
    mode = args.mode

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
    if mode == "dynamic":
        # In dynamic mode, use a continuous timer.
        pygame.time.set_timer(GAME_UPDATE, 1200)
    else:
        # For frozen mode, initialize timing variables.
        last_action_time = pygame.time.get_ticks()
        auto_update_active = False
        last_auto_update_time = 0  # To track when the last auto update was posted.
        previous_block_id = game.current_block.id
	
    CACHE_DIR = "cache/tetris"
    os.makedirs(CACHE_DIR, exist_ok=True)
    state_json_path = os.path.join(CACHE_DIR, "state.json")
    state_path = os.path.join(CACHE_DIR, "screenshot.png")

    while True:
        # Process events.
        for event in pygame.event.get():
            # Save game state on quit, keydown, or GAME_UPDATE events (if game not over).
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN or (event.type == GAME_UPDATE and not game.game_over):
                print("saving state to...")
                print(state_path)
                print(state_json_path)
                pygame.image.save(screen, state_path)
                dump_game_state(game, state_json_path)

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # In frozen mode, update the last action time and cancel auto update.
                if mode == "frozen":
                    last_action_time = pygame.time.get_ticks()
                    auto_update_active = False
                if game.game_over:
                    game.game_over = False
                    game.reset()
                if event.key == pygame.K_LEFT and not game.game_over:
                    game.move_left()
                if event.key == pygame.K_RIGHT and not game.game_over:
                    game.move_right()
                if event.key == pygame.K_DOWN and not game.game_over:
                    game.move_down()
                    game.update_score(0, 1)
                if event.key == pygame.K_UP and not game.game_over:
                    game.rotate()

            # In dynamic mode, the timer-driven GAME_UPDATE event will trigger a move down.
            if event.type == GAME_UPDATE and not game.game_over:
                game.move_down()

        # Frozen mode: Check inactivity and trigger auto GAME_UPDATE events.
        if mode == "frozen" and not game.game_over:
            current_time = pygame.time.get_ticks()
            # If 10 seconds pass without any user input, start auto-updating.
            if not auto_update_active and (current_time - last_action_time >= 10000):
                auto_update_active = True
                last_auto_update_time = current_time
            # If auto update is active, post GAME_UPDATE events at 50 ms intervals.
            if auto_update_active and (current_time - last_auto_update_time >= 25):
                pygame.event.post(pygame.event.Event(GAME_UPDATE))
                last_auto_update_time = current_time

            # If a new block spawns (i.e. current block changes), pause auto updating.
            if game.current_block.id != previous_block_id:
                auto_update_active = False
                last_action_time = current_time
                previous_block_id = game.current_block.id

        # Drawing.
        score_value_surface = title_font.render(str(game.score), True, Colors.white)
        screen.fill(Colors.dark_blue)
        screen.blit(score_surface, (365, 20, 50, 50))
        screen.blit(next_surface, (375, 180, 50, 50))

        if game.game_over:
            screen.blit(game_over_surface, (320, 450, 50, 50))

        pygame.draw.rect(screen, Colors.light_blue, score_rect, 0, 10)
        screen.blit(score_value_surface, score_value_surface.get_rect(centerx=score_rect.centerx,
                                                                       centery=score_rect.centery))
        pygame.draw.rect(screen, Colors.light_blue, next_rect, 0, 10)
        game.draw(screen)

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()
