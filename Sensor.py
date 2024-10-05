from __future__ import annotations

import cv2
import numpy as np

from abc import ABC, abstractmethod
from config import config


class Sensor(ABC):
    def __init__(self, spawn_point, attach_to, callback):
        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        self.actor = self.init_actor(spawn_point, attach_to)
        assert self.actor is not None
        self.actor.listen(callback)

    @abstractmethod
    def init_actor(self, *args, **kwargs):
        pass

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()


class RGBCameraSensor(Sensor):
    def __init__(self, spawn_point, attach_to, callback):
        super().__init__(spawn_point, attach_to, callback)

        self.dist_coeffs = np.zeros((5, 1))
        self.frame_counter = 0

    def init_actor(self, spawn_point, attach_to):
        # Spawn

        sensor_bp = self.client.world.get_blueprint_library().find('sensor.camera.rgb')
        sensor_bp.set_attribute('image_size_x', f'{config.IMAGE_WIDTH}')
        sensor_bp.set_attribute('image_size_y', f'{config.IMAGE_HEIGHT}')
        sensor_bp.set_attribute('fov', f'{config.IMAGE_FOV}')

        sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=attach_to)

        # Calibration

        calibration = np.identity(3)
        # Set the x-coordinate of the principal point
        calibration[0, 2] = config.IMAGE_WIDTH / 2.0
        # Set the y-coordinate of the principal point
        calibration[1, 2] = config.IMAGE_HEIGHT / 2.0
        # Set focal length in the x and y directions
        calibration[0, 0] = calibration[1, 1] = config.IMAGE_WIDTH / (2.0 * np.tan(config.IMAGE_FOV * np.pi / 360.0))

        sensor.calibration = calibration

        return sensor

    @staticmethod
    def numpy_from_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # Drop alpha channel
        array = array[:, :, :3]
        # Convert from BGRA to RGB
        array = array[:, :, ::-1]
        return array

    def undistort_image(self, image):
        if self.actor.calibration is None or self.dist_coeffs is None:
            raise ValueError("Camera calibration and distortion coefficients are not initialized")

        # Get the dimensions of the image
        h, w = image.shape[:2]

        # Compute the optimal new camera matrix (this step is optional, but improves the undistorted image)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.actor.calibration, self.dist_coeffs, (w, h), 1, (w, h))

        # Undistort the image
        undistorted_image = cv2.undistort(image, self.actor.calibration, self.dist_coeffs, None, new_camera_matrix)

        # Crop the image if necessary
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]

        return undistorted_image
