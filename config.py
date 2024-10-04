from dataclasses import dataclass


@dataclass
class DriverConfig:
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_FOV = 120

    FRAME_RATE = 20

    DISPLAY_MANAGER_ROWS = 1
    DISPLAY_MANAGER_COLS = 1


config = DriverConfig()

