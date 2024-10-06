from __future__ import annotations

import pygame
import carla

from Player import ObjectDetectionPlayer, VisualOdometryPlayer
from ClientDisplay import ClientDisplay
from KeyboardController import KeyboardController
from config import config


class CarlaClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CarlaClient, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return

        self.__initialized = True

        self.client = None
        self.world = None
        self.display_manager = None
        self.player = None

    """
    Utils
    """

    def update_spectator_location(self):
        spectator = self.world.get_spectator()

        distance_behind = 5
        vehicle_transform = self.player.actor.get_transform()
        offset = vehicle_transform.get_forward_vector() * -distance_behind

        # noinspection PyArgumentList
        spectator.set_transform(
            carla.Transform(
                # Adjust z for the height above ground
                vehicle_transform.location + offset + carla.Location(z=2.5),
                # Look forward in the direction of the car
                carla.Rotation(pitch=0, yaw=vehicle_transform.rotation.yaw)
            )
        )

    def set_sync_mode(self, mode: bool):
        settings = self.world.get_settings()
        settings.synchronous_mode = mode
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    """
    Run Game Loop
    """

    def run_task(self, Player):
        self.client = carla.Client('127.0.0.1', 2000)
        assert self.client is not None
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        assert self.world is not None

        self.display_manager = ClientDisplay((1, 2))
        self.player = Player()
        assert self.player is not None
        controller = KeyboardController(self.player)

        self.player.actor.set_autopilot(True)

        clock = pygame.time.Clock()
        self.set_sync_mode(True)

        while True:
            self.update_spectator_location()
            self.display_manager.render(self.player.widgets)

            self.world.tick()
            clock.tick_busy_loop(config.FRAME_RATE)

            if controller.parse_quit_events():
                break

        self.set_sync_mode(False)

        self.player.cleanup()
        self.display_manager.cleanup()

    def start(self):
        self.run_task(VisualOdometryPlayer)
        # self.run_task(ObjectDetectionPlayer)
