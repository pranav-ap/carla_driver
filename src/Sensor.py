from __future__ import annotations

import carla
import cv2
import numpy as np

from abc import ABC, abstractmethod
from .config import config


class Sensor(ABC):
    def __init__(self, spawn_point, attach_to, callback):
        from src.CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        self.actor = self._init_actor(spawn_point, attach_to)
        assert self.actor is not None
        self.actor.listen(callback)

    @abstractmethod
    def _init_actor(self, *args, **kwargs):
        pass

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()


class RGBCameraSensor(Sensor):
    def __init__(self, spawn_point, attach_to, callback):
        super().__init__(spawn_point, attach_to, callback)

        self.dist_coeffs = np.zeros((5, 1))
        self.K = self._calibrate()
        self.inv_K = np.linalg.inv(self.K)

    def _init_actor(self, spawn_point, attach_to):
        sensor_bp = self.client.blueprint_library.find('sensor.camera.rgb')
        sensor_bp.set_attribute('image_size_x', f'{config.IMAGE_WIDTH}')
        sensor_bp.set_attribute('image_size_y', f'{config.IMAGE_HEIGHT}')
        sensor_bp.set_attribute('fov', f'{config.IMAGE_FOV}')
        sensor_bp.set_attribute('enable_postprocess_effects', f'{config.IMAGE_NATURAL_NOISE}')

        sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=attach_to)
        return sensor

    def _calibrate(self):
        fov = float(self.actor.attributes['fov'])
        image_width = int(self.actor.attributes['image_size_x'])
        image_height = int(self.actor.attributes['image_size_y'])

        # Calculate focal lengths based on FOV
        fx = image_width / (2.0 * np.tan(np.radians(fov) / 2.0))
        fy = image_height / (2.0 * np.tan(np.radians(fov) / 2.0))

        # Principal points
        cx = image_width / 2.0
        cy = image_height / 2.0

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])

        return K

    def undistort_image(self, image):
        undistorted_image = cv2.undistort(image, self.K, self.dist_coeffs)
        return undistorted_image

    @staticmethod
    def numpy_from_carla_image(image: carla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # Drop alpha channel
        array = array[:, :, :3]
        # Convert from BGR to RGB
        array = array[:, :, ::-1]
        return array
