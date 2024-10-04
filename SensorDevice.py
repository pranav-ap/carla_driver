from __future__ import annotations

import asyncio

import carla
import numpy as np

from typing import Tuple

from config import config


class SensorDevice:
    def __init__(self, sensor_type, spawn_point, sensor_options, callback, display_pos):
        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        self.sensor = self.init_sensor(
            sensor_type,
            spawn_point,
            sensor_options,
            callback
        )

        self.sensor_options = sensor_options
        self.data = self.init_sensor_data(sensor_type)

        self.display_pos: Tuple[int] | None = display_pos
        self.surface = None

        self.client.display_manager.add_sensor(self)

    @staticmethod
    def init_sensor_data(sensor_type):
        if sensor_type == 'sensor.camera.rgb':
            return np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 4))
        elif sensor_type == 'sensor.lidar.ray_cast':
            return np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 4))
        elif sensor_type == 'sensor.other.gnss':
            return [0, 0]
        elif sensor_type == 'sensor.other.imu':
            return {
                'gyro': carla.Vector3D(),
                'accel': carla.Vector3D(),
                'compass': 0
            }
        elif sensor_type == 'sensor.other.collision':
            return False

    def init_sensor(self, sensor_type, spawn_point, sensor_options, callback):
        sensor_bp = self.client.world.get_blueprint_library().find(sensor_type)

        if sensor_type == 'sensor.camera.rgb':
            sensor_bp.set_attribute('image_size_x', f'{config.IMAGE_WIDTH}')
            sensor_bp.set_attribute('image_size_y', f'{config.IMAGE_HEIGHT}')
            sensor_bp.set_attribute('fov', f'{config.IMAGE_FOV}')

            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)

            """
            Calibration
            """

            # calibration = np.identity(3)
            # # Set the x-coordinate of the principal point
            # calibration[0, 2] = config.IMAGE_WIDTH / 2.0
            # # Set the y-coordinate of the principal point
            # calibration[1, 2] = config.IMAGE_HEIGHT / 2.0
            # # Set focal length in the x and y directions
            # calibration[0, 0] = calibration[1, 1] = config.IMAGE_WIDTH / (2.0 * np.tan(config.IMAGE_FOV * np.pi / 360.0))
            #
            # sensor.calibration = calibration

        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_bp.set_attribute('range', '100')
            sensor_bp.set_attribute('dropoff_general_rate', '0.0')
            sensor_bp.set_attribute('dropoff_intensity_limit', '1.0')
            sensor_bp.set_attribute('dropoff_zero_intensity', '0.0')

            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)

        elif sensor_type == 'sensor.other.gnss':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)

        elif sensor_type == 'sensor.other.imu':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)

        elif sensor_type == 'sensor.other.collision':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)

        elif sensor_type == 'sensor.other.lane_invasion':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)

        else:
            return None

        for key in sensor_options:
            sensor_bp.set_attribute(key, sensor_options[key])

        sensor.listen(callback)

        return sensor

    def cleanup(self):
        if self.sensor is not None:
            self.sensor.destroy()

