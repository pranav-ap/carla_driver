from __future__ import annotations

import random
import carla
import cv2
import pygame

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


class RGBCameraSensor(SensorDevice):
    def __init__(self):
        self.surface = None

        self.detection_future = None
        self.matching_future = None

        from DetectionModel import DetectionModel
        self.detector = DetectionModel()
        self.frame_counter = 0

        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_image_color = None

        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2)

        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

    def object_detection(self, image):
        self.frame_counter += 1

        if self.frame_counter % 10 == 0 and (self.detection_future is None or self.detection_future.done()):
            # Submit the detection task to the executor (run it in a separate thread)
            self.detection_future = self.executor.submit(self.detector.pred, image)
            self.frame_counter = 0

        # Check if detection is done and update array with the result
        if self.detection_future and self.detection_future.done():
            try:
                result, image = self.detection_future.result()
                if image is not None and self.client.display_manager.render_enabled():
                    self.surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

            except Exception as e:
                print(f"Detection failed: {e}")

    def visual_odometry(self, prev_image_color, current_image_color):
        prev_image_gray = cv2.cvtColor(prev_image_color, cv2.COLOR_RGB2GRAY)
        current_image_gray = cv2.cvtColor(current_image_color, cv2.COLOR_RGB2GRAY)

        kp1, des1 = self.orb.detectAndCompute(prev_image_gray, None)
        kp2, des2 = self.orb.detectAndCompute(current_image_gray, None)

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        num_matches_to_draw = min(20, len(matches))
        some_matches = matches[:num_matches_to_draw]

        from utils import draw_matches
        matched_image = draw_matches(
            prev_image_color, kp1,
            current_image_color, kp2,
            some_matches
        )

        return matched_image

    def rgb_camera_visual_odometry(self, current_image_color):
        self.frame_counter += 1

        if self.frame_counter % 5 == 0 and self.prev_image_color is not None and (self.matching_future is None or self.matching_future.done()):
            self.matching_future = self.executor.submit(
                self.visual_odometry,
                self.prev_image_color,
                current_image_color
            )

            self.frame_counter = 0

        if self.matching_future and self.matching_future.done():
            try:
                latest_matched_image = self.matching_future.result()
                if latest_matched_image is not None and self.client.display_manager.render_enabled():
                    self.surface = pygame.surfarray.make_surface(latest_matched_image.swapaxes(0, 1))

            except Exception as e:
                print(f"Detection failed: {e}")

        self.prev_image_color = current_image_color


class EgoCar:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EgoCar, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return

        self.__initialized = True

        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.actor = self._setup_actor()

        self.rgb_camera_front: SensorDevice = SensorDevice(
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=1.5, z=2.1), carla.Rotation(yaw=0)),
            sensor_options={},
            callback=self.rgb_camera_front_callback,
            display_pos=[0, 0]
        )

        self.rgb_camera_left: SensorDevice | None = None
        self.rgb_camera_right: SensorDevice | None = None
        self.rgb_camera_rear: SensorDevice | None = None

        """
        self.rgb_camera_rear = SensorDevice(
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=-1.5, z=2.1), carla.Rotation(yaw=180)),
            sensor_options={},
            display_pos=None  # None [0, 1]
        )

        self.rgb_camera_left = SensorDevice(
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=0, y=-1.0, z=2.1), carla.Rotation(yaw=-90)),
            sensor_options={},
            display_pos=None  # None [0, 0]
        )

        self.rgb_camera_right = SensorDevice(
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=0, y=1.0, z=2.1), carla.Rotation(yaw=90)),
            sensor_options={},
            display_pos=None  # None [0, 2]
        )
        """

    """
    Utils
    """

    def _setup_actor(self):
        bp = self.client.world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        spawn_point = random.choice(self.client.world.get_map().get_spawn_points())
        actor = self.client.world.spawn_actor(bp, spawn_point)

        return actor

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()

    """
    Sensor Callbacks
    """

    def rgb_camera_front_callback(self, image):
        from utils import rgb_camera_numpy_from_image
        array = rgb_camera_numpy_from_image(image)
        # self.rgb_camera_object_detection(array)
        self.rgb_camera_visual_odometry(array)
