import colorsys
from typing import Optional

import numpy as np
import pygame


class Renderer:
    def __init__(
            self,
            num_rows: int,
            num_cols: int,
            num_colours: int,
            num_moves: int,
            render_fps: int,
            render_mode: Optional[str] = "human"
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colours = num_colours
        self.num_moves = num_moves
        self.render_fps = render_fps
        self.render_mode = render_mode

        self.screen = None
        self.width = None
        self.height = None
        self.colour_map = []
        for i in range(1, num_colours + 1):  # Skip white
            saturation = 0.6
            lightness = 0.5
            hue = i / num_colours
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            rgb = tuple(int(val * 255) for val in rgb)  # Scale to [0, 255]
            self.colour_map.append(rgb)

    def render(self, board: np.ndarray, moves_left: int) -> np.ndarray:
        if self.screen is None:
            self._init_pygame()
        white = (255, 255, 255)
        black = (0, 0, 0)
        self.screen.fill(white)

        board_x = (self.screen_width - self.board_render_width) / 2
        board_y = self.text_area_height + 3 * self.margin_size
        # Draw the board
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                tile_colour = board[0, row, col]
                tile_type = board[1, row, col]

                if tile_colour == 0:
                    color = black
                else:
                    color = self.colour_map[tile_colour - 1]

                x = board_x + col * (self.tile_size + self.spacing)
                y = board_y + row * (self.tile_size + self.spacing)
                # assert 0 <= x <= self.screen_width - self.tile_size - self.margin_size
                # assert 0 <= y <= self.screen_height - self.tile_size - self.margin_size

                if tile_type > 0:  # Coloured tiles.
                    pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))
                if tile_type == 2:  # Vertical laser
                    pygame.draw.rect(self.screen, black,
                                     (x + self.tile_size / 3, y, self.tile_size / 3, self.tile_size))
                elif tile_type == 3:  # Horizontal laser
                    pygame.draw.rect(self.screen, black,
                                     (x, y + self.tile_size / 3, self.tile_size, self.tile_size / 3))
                elif tile_type == 4:  # Bomb
                    pygame.draw.polygon(self.screen, black, [
                        (x + self.tile_size / 2, y),
                        (x + self.tile_size, y + self.tile_size / 2),
                        (x + self.tile_size / 2, y + self.tile_size),
                        (x, y + self.tile_size / 2)
                    ])
                elif tile_type == -1:  # Cookie
                    pygame.draw.circle(self.screen, black, (x + self.tile_size / 2, y + self.tile_size / 2),
                                       self.tile_size / 3)

        # Display moves left at the top in the center

        font = pygame.font.SysFont("helvetica", self.font_size)
        text_surface = font.render(f"Moves Left: {moves_left}", True, black)
        text_x = (self.screen_width - text_surface.get_width()) / 2
        text_y = (self.text_area_height - text_surface.get_height()) / 1.8
        self.screen.blit(text_surface, (text_x, text_y))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)
        else:  # mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)).copy()


    def _init_pygame(self):

        pygame.init()
        info = pygame.display.Info()
        board_screen_ratio = 0.35
        if self.num_cols > self.num_rows:
            self.board_render_width = info.current_w * board_screen_ratio
            self.board_render_height = np.ceil(self.board_render_width * (self.num_rows / self.num_cols))
        else:
            self.board_render_height = info.current_h * board_screen_ratio
            self.board_render_width = np.ceil(self.board_render_height * (self.num_cols / self.num_rows))

        margin_ratio = 0.02  # Proportion of board size to add on for each margin
        self.text_area_height = 40

        # Available height and width for the board after removing margins and text area
        self.margin_size = np.round(min(self.board_render_width, self.board_render_height) * margin_ratio, 1)
        spacing_ratio = 0.04  # Proportion of tile_size to be used for spacing
        max_tile_width = self.board_render_width / self.num_cols
        max_tile_height = self.board_render_height / self.num_rows
        self.tile_size = np.floor((1 - spacing_ratio) * min(max_tile_width, max_tile_height))
        self.spacing = np.round((min(max_tile_width, max_tile_height) * spacing_ratio), 2)
        # assert self.tile_size > 10, self.tile_size

        self.board_render_width = (self.tile_size + self.spacing) * self.num_cols
        self.board_render_height = (self.tile_size + self.spacing) * self.num_rows
        self.screen_width = self.board_render_width + 2 * self.margin_size
        self.screen_height = self.board_render_height + 4 * self.margin_size + self.text_area_height

        # assert self.board_render_width + (2 * self.margin_size) <= self.screen_width
        # assert self.board_render_height + (4 * self.margin_size) + self.text_area_height <= self.screen_height
        #
        # assert self.screen_width >= self.board_render_width + 2 * self.margin_size
        # assert self.screen_height >= self.board_render_height + 4 * self.margin_size + self.text_area_height

        self.font_size = (self.text_area_height * 8) // 10
        font = pygame.font.SysFont("helvetica", self.font_size)
        max_text = font.render(f"Moves Left: {self.num_moves}", True, (0, 0, 0))
        min_width = max_text.get_width() + 2 * self.margin_size

        self.screen_width = max(self.screen_width, min_width)
        self.board_render_width *= 0.9
        self.board_render_height *= 0.9
        self.margin_size *=0.9
        self.text_area_height *= 0.9
        self.tile_size *= 0.9
        self.spacing *= 0.9
        # assert self.screen_width <= info.current_w
        # assert self.screen_height <= info.current_h
        if self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Tile Match")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))  # Visible window
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

