from __future__ import annotations

import random
import carla
import cv2
import pygame

from SensorDevice import SensorDevice


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

        self.rgb_camera_front: SensorDevice | None = None
        self.rgb_camera_left: SensorDevice | None = None
        self.rgb_camera_right: SensorDevice | None = None
        self.rgb_camera_rear: SensorDevice | None = None
        self.lidar_sensor: SensorDevice | None = None

        self.actor = None

        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.detection_future = None
        self.matching_future = None

        from DetectionModel import DetectionModel
        self.detector = DetectionModel()
        self.frame_counter = 0

        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_image_color = None

        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

    def setup(self):
        self._setup_actor()
        self._setup_sensors()

    def _setup_actor(self):
        bp = self.client.world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        spawn_point = random.choice(self.client.world.get_map().get_spawn_points())
        self.actor = self.client.world.spawn_actor(bp, spawn_point)

    def _setup_sensors(self):
        self.rgb_camera_front = SensorDevice(
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=1.5, z=2.1), carla.Rotation(yaw=0)),
            sensor_options={},
            callback=self.rgb_camera_front_callback,
            display_pos=[0, 0]
        )

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

        self.lidar_sensor = SensorDevice(
            'sensor.lidar.ray_cast',
            carla.Transform(carla.Location(z=2.5), carla.Rotation(pitch=-15, yaw=90)),
            sensor_options={
                'channels': '64',
                'range': '100',
                'points_per_second': '56000',
                'rotation_frequency': '10'
            },
            display_pos=[0, 1]
        )
        """

    """
    Sensor Callbacks Utils
    """

    def rgb_camera_object_detection(self, image):
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
                    self.rgb_camera_front.surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

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
            self.matching_future = self.executor.submit(self.visual_odometry, self.prev_image_color, current_image_color)
            self.frame_counter = 0

        if self.matching_future and self.matching_future.done():
            try:
                latest_matched_image = self.matching_future.result()
                if latest_matched_image is not None and self.client.display_manager.render_enabled():
                    self.rgb_camera_front.surface = pygame.surfarray.make_surface(latest_matched_image.swapaxes(0, 1))

            except Exception as e:
                print(f"Detection failed: {e}")

        self.prev_image_color = current_image_color

    """
    Sensor Callbacks
    """

    def rgb_camera_front_callback(self, image):
        from utils import rgb_camera_numpy_from_image
        array = rgb_camera_numpy_from_image(image)
        # self.rgb_camera_object_detection(array)
        self.rgb_camera_visual_odometry(array)

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()
