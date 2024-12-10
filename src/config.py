from dataclasses import dataclass


@dataclass
class DriverConfig:
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_FOV = 120
    IMAGE_NATURAL_NOISE = True

    FRAME_RATE = 20


config = DriverConfig()

