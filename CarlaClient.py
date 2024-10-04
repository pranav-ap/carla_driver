from __future__ import annotations

import pygame
import carla

from pygame.locals import K_ESCAPE, K_SPACE
from pygame.locals import K_a, K_d, K_s, K_w

from DisplayManager import DisplayManager
from EgoCar import EgoCar
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

        self.car: EgoCar | None = None
        self.display_manager: DisplayManager | None = None

    """
    Manual Control
    """

    def control(self):
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = self.car.actor.get_control()
        self.handle_throttle(control, keys)
        self.handle_steering(control, keys)
        control.hand_brake = keys[K_SPACE]

        self.car.actor.apply_control(control)
        return False

    @staticmethod
    def handle_throttle(control, keys):
        control.throttle = 1 if keys[K_w] or keys[K_s] else 0
        control.reverse = keys[K_s]

    @staticmethod
    def handle_steering(control, keys):
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0

    """
    Game Loop
    """

    def __enter__(self):
        self.client = carla.Client('127.0.0.1', 2000)
        assert self.client is not None
        self.client.set_timeout(10.0)

        """
        World
        """

        self.world = self.client.get_world()
        assert self.world is not None

        self.set_synchronous_mode(True)

        """
        Display Manager
        """

        self.display_manager = DisplayManager(
            grid_size=[config.DISPLAY_MANAGER_ROWS, config.DISPLAY_MANAGER_COLS],
        )

        """
        Car
        """

        self.car = EgoCar()
        self.car.setup()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.car is not None:
            self.car.cleanup()

        if self.display_manager is not None:
            self.display_manager.cleanup()

        self.set_synchronous_mode(False)

    def update_spectator_location(self):
        spectator = self.world.get_spectator()

        distance_behind = 5
        vehicle_transform = self.car.actor.get_transform()
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

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def game_loop(self):
        pygame_clock = pygame.time.Clock()

        while True:
            self.world.tick()
            self.update_spectator_location()
            pygame_clock.tick_busy_loop(config.FRAME_RATE)

            self.display_manager.render()

            pygame.event.pump()

            # Check if the window is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    return

            if self.control():
                return
