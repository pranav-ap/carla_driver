from __future__ import annotations
from utils.logger_setup import logger

import random
import pygame
import carla

from ClientDisplay import ClientDisplay
from Player import VisualOdometryPlayer
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

        self.client = carla.Client('localhost', 2000)
        assert self.client is not None
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        assert self.world is not None

        self.spectator = self.world.get_spectator()
        assert self.spectator is not None

        self.blueprint_library = self.world.get_blueprint_library()
        assert self.blueprint_library is not None

        self.player = None
        self.client_display = None

    """
    Utils
    """

    def update_spectator_location(self):
        distance_behind = 5
        vehicle_transform = self.player.actor.get_transform()
        offset = vehicle_transform.get_forward_vector() * -distance_behind

        # noinspection PyArgumentList
        self.spectator.set_transform(
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

    def set_weather(self):
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=10.0,
            sun_azimuth_angle=90.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
        )

        self.world.set_weather(weather)

    def spawn_npc(self, count):
        spawn_points = self.world.get_map().get_spawn_points()

        for i in range(count):
            vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle'))
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        for v in self.world.get_actors().filter('*vehicle*'):
            v.set_autopilot(True)

        self.player.actor.set_autopilot(False)

        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

    """
    Run Game Loop
    """

    def run_vo(self):
        self.client_display = ClientDisplay((1, 2))

        self.player = VisualOdometryPlayer()
        assert self.player is not None
        self.player.actor.set_autopilot(True)

        from KeyboardController import KeyboardController
        controller = KeyboardController(self.player)

        self.set_sync_mode(True)
        clock = pygame.time.Clock()

        while True:
            self.client_display.render(self.player.widgets)
            self.update_spectator_location()

            self.world.tick()
            # Limit the game loop to run at X frames per second
            # clock.tick_busy_loop(config.FRAME_RATE)
            clock.tick(config.FRAME_RATE)

            if controller.parse_events(self.player.actor):
                break

        self.set_sync_mode(False)

        self.client_display.cleanup()
        self.player.cleanup()

