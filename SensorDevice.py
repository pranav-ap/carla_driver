from __future__ import annotations

import carla
import pygame
import numpy as np

from typing import Dict, Any, List, Tuple

from config import config


class SensorDevice:
    def __init__(self, sensor_type, spawn_point, sensor_options, display_pos):
        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        self.sensor = self.init_sensor(
            sensor_type,
            spawn_point,
            sensor_options
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

    def init_sensor(self, sensor_type, spawn_point, sensor_options):
        sensor_bp = self.client.world.get_blueprint_library().find(sensor_type)

        if sensor_type == 'sensor.camera.rgb':
            sensor_bp.set_attribute('image_size_x', f'{config.IMAGE_WIDTH}')
            sensor_bp.set_attribute('image_size_y', f'{config.IMAGE_HEIGHT}')
            sensor_bp.set_attribute('fov', f'{config.IMAGE_FOV}')

            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)
            sensor.listen(self.rgb_callback)

            """
            Calibration
            """

            calibration = np.identity(3)
            # Set the x-coordinate of the principal point
            calibration[0, 2] = config.IMAGE_WIDTH / 2.0
            # Set the y-coordinate of the principal point
            calibration[1, 2] = config.IMAGE_HEIGHT / 2.0
            # Set focal length in the x and y directions
            calibration[0, 0] = calibration[1, 1] = config.IMAGE_WIDTH / (2.0 * np.tan(config.IMAGE_FOV * np.pi / 360.0))

            sensor.calibration = calibration

        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_bp.set_attribute('range', '100')
            sensor_bp.set_attribute('dropoff_general_rate', '0.0')
            sensor_bp.set_attribute('dropoff_intensity_limit', '1.0')
            sensor_bp.set_attribute('dropoff_zero_intensity', '0.0')

            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)
            sensor.listen(self.lidar_callback)

        elif sensor_type == 'sensor.other.gnss':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)
            sensor.listen(self.gnss_callback())

        elif sensor_type == 'sensor.other.imu':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)
            sensor.listen(self.imu_callback())

        elif sensor_type == 'sensor.other.collision':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)
            sensor.listen(self.collision_callback())

        elif sensor_type == 'sensor.other.lane_invasion':
            sensor = self.client.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.client.car.actor)
            sensor.listen(self.lane_inv_callback())

        else:
            return None

        for key in sensor_options:
            sensor_bp.set_attribute(key, sensor_options[key])

        return sensor

    """
    Sensor Callbacks
    """

    def rgb_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.data = array

        if self.client.display_manager.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def depth_callback(self, image):
        image.convert(carla.ColorConverter.LogarithmicDepth)

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.data = array

        if self.client.display_manager.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def lidar_callback(self, image):
        disp_size = client.display_manager.get_display_size()

        lidar_range = 2.0 * float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))

        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros(lidar_img_size, dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        self.data = lidar_img

        if self.client.display_manager.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

    def gnss_callback(self, data):
        self.data = [data.latitude, data.longitude]

    def imu_callback(self, data):
        self.sensor.data = {
            'gyro': data.gyroscope,
            'accel': data.accelerometer,
            'compass': data.compass,
        }

    def lane_inv_callback(self, data):
        pass

    def collision_callback(self, data):
        pass

    def cleanup(self):
        if self.sensor is not None:
            self.sensor.destroy()

