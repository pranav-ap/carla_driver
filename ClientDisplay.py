from __future__ import annotations

import pygame
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from config import config


@dataclass
class PyGameWidget:
    grid_start_position: Tuple[int, int]
    span: Tuple[int, int]
    surface: Optional[pygame.Surface] = None
    new_size: Optional[Tuple[int, int]] = None
    offset: Optional[Tuple[int, int]] = None


class ClientDisplay:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ClientDisplay, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, grids_shape):
        if self.__initialized:
            return

        self.__initialized = True

        self._init_pygame()

        self.grids_shape = grids_shape  # rows, cols

        self.requires_resize = False
        self.single_grid_size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        self.single_grid_size = self._scale_down_single_grid_size()

        self.window_size = self._init_window_size()

        self.display = pygame.display.set_mode(self.window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        assert self.display is not None

    """
    Setup & Cleanup
    """

    @staticmethod
    def _init_pygame():
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("CARLA Driver")

    @staticmethod
    def cleanup():
        pygame.quit()

    def _scale_down_single_grid_size(self):
        rows, cols = self.grids_shape
        orig_w, orig_h = self.single_grid_size

        max_window_width = pygame.display.get_desktop_sizes()[0][0]
        max_window_height = pygame.display.get_desktop_sizes()[0][1]

        max_w = max_window_width // cols
        max_h = max_window_height // rows

        # If no scaling is necessary, retain the original size
        new_w = orig_w
        new_h = orig_h

        # Scale down based on the most restrictive dimension (width or height)
        if orig_w > max_w or orig_h > max_h:
            self.requires_resize = True

            # Calculate the width and height scale factors
            width_scale_factor = max_w / orig_w
            height_scale_factor = max_h / orig_h

            # Use the smaller scale factor to maintain the aspect ratio
            scale_factor = min(width_scale_factor, height_scale_factor)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)

        return new_w, new_h

    def _init_window_size(self):
        rows, cols = self.grids_shape
        w, h = self.single_grid_size
        window_size = (w * cols, h * rows)
        return window_size

    """
    Render
    """

    def render(self, screens: Dict[str, PyGameWidget]):
        for name, w in screens.items():
            if w.surface is None:
                continue

            if w.offset is None:
                w.offset = self._get_surface_offset(w)

            if not self.requires_resize:
                self.display.blit(w.surface, w.offset)
                continue

            if w.new_size is None:
                w.new_size = self._get_surface_new_size(w)

            scaled_surface = pygame.transform.scale(w.surface, w.new_size)
            self.display.blit(scaled_surface, w.offset)

        pygame.display.flip()

    def _get_surface_offset(self, w: PyGameWidget):
        row, col = w.grid_start_position
        single_grid_w, single_grid_h = self.single_grid_size

        offset_x = col * single_grid_w
        offset_y = row * single_grid_h

        return offset_x, offset_y

    def _get_surface_new_size(self, w: PyGameWidget):
        row_span, col_span = w.span
        single_grid_w, single_grid_h = self.single_grid_size

        total_width = col_span * single_grid_w
        total_height = row_span * single_grid_h

        return total_width, total_height
