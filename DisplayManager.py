from __future__ import annotations

from typing import Dict, Tuple

import pygame

from abc import ABC, abstractmethod
from dataclasses import dataclass
from config import config


@dataclass
class PyGameScreen:
    surface: pygame.Surface | None
    grid_position: Tuple[int, int]


class DisplayManager(ABC):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DisplayManager, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return

        self.__initialized = True

        self._init_pygame()

        self.grid_size = self._init_grid_size()  # [rows, cols]
        self.new_image_size = self._init_new_image_size()
        self.requires_scaling = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT) != self.new_image_size
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

    @abstractmethod
    def _init_grid_size(self):
        pass

    def _init_new_image_size(self):
        original_image_width, original_image_height = config.IMAGE_WIDTH, config.IMAGE_HEIGHT

        # Primary monitor
        max_window_width = pygame.display.get_desktop_sizes()[0][0]

        # Calculate the new width for each image
        max_image_width = max_window_width // self.grid_size[1]  # Width for each column
        new_image_width = min(original_image_width, max_image_width)

        # Ensure height scaling proportionally
        scale_factor = new_image_width / original_image_width
        new_image_height = int(original_image_height * scale_factor)

        return new_image_width, new_image_height

    @abstractmethod
    def _init_window_size(self):
        pass

    @staticmethod
    def cleanup():
        pygame.quit()

    """
    Render
    """

    def render(self, screens: Dict[str, PyGameScreen]):
        for name, s in screens.items():
            if s.surface is None:
                continue

            offset = self.get_display_offset(s.grid_position)

            if self.requires_scaling:
                w, h = self.window_size
                scaled_surface = pygame.transform.scale(s.surface, (w // self.grid_size[1], h // self.grid_size[0]))
                self.display.blit(scaled_surface, offset)
            else:
                self.display.blit(s.surface, offset)

        pygame.display.flip()

    def get_display_offset(self, grid_position):
        # Calculate the offset based on the grid position
        row, col = grid_position
        new_image_width = self.window_size[0] // self.grid_size[1]
        new_image_height = self.window_size[1] // self.grid_size[0]

        return col * new_image_width, row * new_image_height


class VisualOdometryDisplayManager(DisplayManager):
    def __init__(self):
        super().__init__()

    """
    Setup & Cleanup
    """

    def _init_grid_size(self):
        return [1, 1]

    def _init_window_size(self):
        w, h = self.new_image_size
        window_size = (w * self.grid_size[1] * 2, h * self.grid_size[0])
        return window_size


class ObjectDetectionDisplayManager(DisplayManager):
    def __init__(self):
        super().__init__()

    """
    Setup & Cleanup
    """

    def _init_grid_size(self):
        return [1, 1]

    def _init_window_size(self):
        w, h = self.new_image_size
        window_size = (w * self.grid_size[1], h * self.grid_size[0])
        return window_size
