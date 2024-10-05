from __future__ import annotations

import random
import carla
import pygame

from abc import ABC, abstractmethod
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from Sensor import RGBCameraSensor
from DisplayManager import PyGameScreen
from FeatueMatcher import FeatureMatcher
from ObjectDetector import ObjectDetector


class PlayerActions:
    @staticmethod
    def visual_odometry(player, sensor_name: str, screen_name: str, current_color_image):
        sensor = player.sensors[sensor_name]
        sensor.frame_counter += 1

        if sensor.frame_counter % 3 == 0 and player.previous_image is not None and (player.feature_matcher_future is None or player.feature_matcher_future.done()):
            player.feature_matcher_future = player.executor.submit(
                player.feature_matcher.predict,
                player.previous_image,
                current_color_image
            )

            sensor.frame_counter = 0

        if player.feature_matcher_future and player.feature_matcher_future.done():
            try:
                latest_matched_image = player.feature_matcher_future.result()
                if latest_matched_image is not None:
                    screen = player.screens[screen_name]
                    screen.surface = pygame.surfarray.make_surface(latest_matched_image.swapaxes(0, 1))

            except Exception as e:
                print(f"Feature Matching Exception : {e}")

        player.previous_image = current_color_image

    @staticmethod
    def object_detection(player: ObjectDetectionPlayer, sensor_name: str, screen_name: str, current_color_image):
        sensor = player.sensors[sensor_name]
        sensor.frame_counter += 1

        if sensor.frame_counter % 5 == 0 and (player.object_detector_future is None or player.object_detector_future.done()):
            player.object_detector_future = player.executor.submit(
                player.object_detector.predict,
                current_color_image
            )

            sensor.frame_counter = 0

        if player.object_detector_future and player.object_detector_future.done():
            try:
                result, latest_boxed_image = player.object_detector_future.result()
                if latest_boxed_image is not None:
                    screen = player.screens[screen_name]
                    screen.surface = pygame.surfarray.make_surface(latest_boxed_image.swapaxes(0, 1))

            except Exception as e:
                print(f"Object Detection Exception : {e}")


class Player(ABC):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Player, cls).__new__(cls)
            cls._instance.__initialized = False

        return cls._instance

    def __init__(self, max_workers):
        if self.__initialized:
            return

        self.__initialized = True

        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.actor = self._init_actor()
        assert self.actor is not None

        self.sensors: Dict[str, RGBCameraSensor] = self._init_sensors()
        self.screens: Dict[str, PyGameScreen] = self._init_surfaces()

    """
    Setup & Cleanup
    """

    def _init_actor(self):
        world = self.client.world

        bp = world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Try to spawn the actor in a free spawn point
        for spawn_point in spawn_points:
            actor = world.try_spawn_actor(bp, spawn_point)
            return actor

        print("Failed to spawn actor - all spawn points are occupied.")
        exit(1)

    @abstractmethod
    def _init_sensors(self):
        pass

    @abstractmethod
    def _init_surfaces(self):
        pass

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()

        for name, s in self.sensors.items():
            s.cleanup()

        self.executor.shutdown(wait=True)


class VisualOdometryPlayer(Player):
    def __init__(self):
        super().__init__(max_workers=1)

        self.feature_matcher = FeatureMatcher()
        self.feature_matcher_future = None
        self.previous_image = None

    """
    Setup & Cleanup
    """

    def _init_sensors(self):
        sensors = dict()

        sensors['RGB Camera Front'] = RGBCameraSensor(
            spawn_point=carla.Transform(
                carla.Location(x=1.5, z=2.1),
                carla.Rotation(yaw=0)
            ),
            attach_to=self.actor,
            callback=self.rgb_camera_front_callback
        )

        return sensors

    def _init_surfaces(self):
        screens = dict()
        screens['VO Screen'] = PyGameScreen(None, (0, 0))
        return screens

    """
    Act
    """

    def rgb_camera_front_callback(self, image):
        image = RGBCameraSensor.numpy_from_image(image)
        image = self.sensors['RGB Camera Front'].undistort_image(image)
        PlayerActions.visual_odometry(self, 'RGB Camera Front', 'VO Screen', image)


class ObjectDetectionPlayer(Player):
    def __init__(self):
        super().__init__(max_workers=3)

        self.object_detector = ObjectDetector()
        self.object_detector_future = None

    """
    Setup & Cleanup
    """

    def _init_sensors(self):
        sensors = dict()

        sensors['RGB Camera Front'] = RGBCameraSensor(
            spawn_point=carla.Transform(
                carla.Location(x=1.5, z=2.1),
                carla.Rotation(yaw=0)
            ),
            attach_to=self.actor,
            callback=self.rgb_camera_front_callback
        )

        return sensors

    def _init_surfaces(self):
        screens = dict()
        screens['OD Screen'] = PyGameScreen(None, (0, 0))
        return screens

    """
    Act
    """

    def rgb_camera_front_callback(self, image):
        image = RGBCameraSensor.numpy_from_image(image)
        image = self.sensors['RGB Camera Front'].undistort_image(image)
        try:
            PlayerActions.object_detection(self, 'RGB Camera Front', 'OD Screen', image)
        except AttributeError as e:
            print(f'OD attribute error again! {e}')
