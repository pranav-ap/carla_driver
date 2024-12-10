from __future__ import annotations
from utils.logger_setup import logger

import random
import carla
import numpy as np
import pygame

from abc import ABC, abstractmethod
from typing import Dict, Optional
from .Sensor import RGBCameraSensor
from .ClientDisplay import PyGameWidget
from .FeatueMatcher import FeatureMatcher


class Player(ABC):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Player, cls).__new__(cls)
            cls._instance.__initialized = False

        return cls._instance

    def __init__(self):
        if self.__initialized:
            return

        self.__initialized = True

        from src.CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        self.actor = self._init_actor()
        assert self.actor is not None

        self.sensors: Dict[str, RGBCameraSensor] = self._init_sensors()
        self.widgets: Dict[str, PyGameWidget] = self._init_widgets()

    """
    Setup & Cleanup
    """

    def _init_actor(self):
        bp = self.client.blueprint_library.filter('vehicle.audi.tt')[0]

        if bp.has_attribute('color'):
            color = bp.get_attribute('color').recommended_values[0]
            bp.set_attribute('color', color)

        spawn_points = self.client.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Try to spawn the actor in a free spawn point
        for spawn_point in spawn_points:
            actor = self.client.world.try_spawn_actor(bp, spawn_point)
            return actor

        logger.warning("Failed to spawn actor - all spawn points are occupied.")
        exit(1)

    @abstractmethod
    def _init_sensors(self):
        pass

    @abstractmethod
    def _init_widgets(self):
        pass

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()

        for name, s in self.sensors.items():
            s.cleanup()


class VisualOdometryPlayer(Player):
    def __init__(self):
        super().__init__()

        self.old_frame: Optional[carla.Image] = None
        self.new_frame: Optional[carla.Image] = None

        self.feature_matcher = FeatureMatcher()

        self.pose_estimate = None
        self.gt_path = []
        self.estimated_path = []

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
            callback=lambda image: self.save_latest_frame(image)
        )

        return sensors

    def _init_widgets(self):
        widgets = dict()

        widgets['VO Screen'] = PyGameWidget(
            grid_start_position=(0, 0),  # row, col
            span=(1, 2)  # row, col
        )

        return widgets

    def visualize_estimate(self):
        from utils import visualize_paths2
        visualize_paths2(
            self.gt_path,
            self.estimated_path,
            "Visual Odometry",
            file_out=f"../output/plot.html"
        )

    """
    Sensor Callbacks
    """

    def save_latest_frame(self, image: carla.Image):
        if self.old_frame is None:
            self.old_frame = image
            return

        self.new_frame = image

    """
    Act
    """

    def act(self):
        self.visual_odometry('RGB Camera Front')

    def visual_odometry(self, sensor_name: str):
        if self.old_frame is None or self.new_frame is None:
            return

        old_frame = RGBCameraSensor.numpy_from_carla_image(self.old_frame)
        new_frame = RGBCameraSensor.numpy_from_carla_image(self.new_frame)
        self.old_frame = self.new_frame

        K = self.sensors[sensor_name].K
        matched_image, T, repeatability_score = self.feature_matcher.predict(
            old_frame,
            new_frame,
            K
        )

        if matched_image is None:
            return

        gt_pose: carla.Transform = self.actor.get_transform()

        if self.pose_estimate is None:
            self.pose_estimate = gt_pose.get_inverse_matrix()

        self.pose_estimate = self.pose_estimate @ np.linalg.inv(T)

        self.gt_path.append((gt_pose.location.x, gt_pose.location.y))
        self.estimated_path.append((self.pose_estimate[0, 3], self.pose_estimate[1, 3]))

        self.handle_ui(matched_image, repeatability_score)

    def handle_ui(self, matched_image, repeatability_score):
        print(f"repeatability_score : {repeatability_score:.3f}")

        widget = self.widgets['VO Screen']
        widget.surface = pygame.surfarray.make_surface(matched_image.swapaxes(0, 1))
