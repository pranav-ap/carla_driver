from __future__ import annotations

import asyncio
import random
import carla
import cv2
import pygame
import numpy as np
from PIL import Image

from SensorDevice import SensorDevice
from detection_model import DetectionModel


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
        self.depth_camera: SensorDevice | None = None
        self.gnss_sensor: SensorDevice | None = None
        self.imu_sensor: SensorDevice | None = None

        self.actor = None

        self.detector = DetectionModel()
        self.frame_counter = 0
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.detection_future = None  # Store future results

        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None

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
    Viz Utils
    """

    @staticmethod
    def draw_matches(img1, kp1, img2, kp2, matches, orientation='vertical'):
        # Convert images to BGR format if they are grayscale
        if len(img1.shape) == 2:  # Grayscale to BGR
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:  # Grayscale to BGR
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        img1 = self.put_text(img1, "top_center", "Previous Frame")
        img2 = self.put_text(img2, "top_center", "Current Frame")

        # Create a new image for stacking
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        if orientation == 'vertical':
            combined_height = height1 + height2
            combined_width = max(width1, width2)
            combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined_img[:height1, :width1] = img1
            combined_img[height1:combined_height, :width2] = img2

            # Adjust keypoints for the second image
            adjusted_kp2 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + height1, kp.size) for kp in kp2]
        elif orientation == 'horizontal':
            combined_height = max(height1, height2)
            combined_width = width1 + width2
            combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined_img[:height1, :width1] = img1
            combined_img[:height2, width1:combined_width] = img2

            # Adjust keypoints for the second image
            adjusted_kp2 = [cv2.KeyPoint(kp.pt[0] + width1, kp.pt[1], kp.size) for kp in kp2]
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'")

        # Draw matches on the combined image
        for match in matches:
            pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
            pt2 = (int(adjusted_kp2[match.trainIdx].pt[0]), int(adjusted_kp2[match.trainIdx].pt[1]))

            # Draw lines between the matched keypoints
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(combined_img, pt1, pt2, color, 2)

            # Draw circles around keypoints
            cv2.circle(combined_img, pt1, 5, (255, 255, 255), -1)  # Keypoint in first image
            cv2.circle(combined_img, pt2, 5, (255, 255, 255), -1)  # Keypoint in second image

        return combined_img

    @staticmethod
    def put_text(image, org, text, color=(0, 0, 255), fontScale=0.7, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        if not isinstance(org, tuple):
            (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
            org_w = 0
            org_h = 0

            h, w, *_ = image.shape

            place_h, place_w = org.split("_")

            if place_h == "top":
                org_h = label_height
            elif place_h == "bottom":
                org_h = h
            elif place_h == "center":
                org_h = h // 2 + label_height // 2

            if place_w == "left":
                org_w = 0
            elif place_w == "right":
                org_w = w - label_width
            elif place_w == "center":
                org_w = w // 2 - label_width // 2

            org = (org_w, org_h)

        image = cv2.putText(image, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        return image

    @staticmethod
    def rgb_camera_numpy_from_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # Drop alpha channel
        array = array[:, :, :3]
        # Convert from BGRA to RGB
        array = array[:, :, ::-1]
        return array

    """
    Sensor Callbacks Utils
    """

    def rgb_camera_object_detection(self, array):
        self.frame_counter += 1

        if self.frame_counter % 10 == 0 and (self.detection_future is None or self.detection_future.done()):
            # Submit the detection task to the executor (run it in a separate thread)
            self.detection_future = self.executor.submit(self.detector.pred, array)
            self.frame_counter = 0

        # Check if detection is done and update array with the result
        if self.detection_future and self.detection_future.done():
            try:
                r, array = self.detection_future.result()  # Get the result from the detection
            except Exception as e:
                print(f"Detection failed: {e}")

        return array

    def visual_odometry(self, prev_frame, curr_frame):
        kp1, des1 = self.orb.detectAndCompute(prev_frame, None)
        kp2, des2 = self.orb.detectAndCompute(curr_frame, None)

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        num_matches_to_draw = min(10, len(matches))
        some_matches = matches[:num_matches_to_draw]

        matched_image = self.draw_matches(
            prev_frame, kp1,
            curr_frame, kp2,
            some_matches,
            orientation='vertical'
        )

        return matched_image

    def rgb_camera_visual_odometry(self, array):
        # Convert to grayscale for feature detection
        gray_frame = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

        if self.frame_counter > 0 and self.prev_frame is not None:
            array = self.visual_odometry(self.prev_frame, gray_frame)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        self.prev_frame = gray_frame
        self.frame_counter += 1

        return array

    """
    Sensor Callbacks
    """

    def rgb_camera_front_callback(self, image):
        array = self.rgb_camera_numpy_from_image(image)
        # array = self.rgb_camera_object_detection(array)
        array = self.rgb_camera_visual_odometry(array)

        if self.client.display_manager.render_enabled():
            self.rgb_camera_front.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def depth_callback(self, image):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        array = self.rgb_camera_numpy_from_image(image)

        if self.client.display_manager.render_enabled():
            self.depth_camera.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def lidar_callback(self, image):
        disp_size = self.client.display_manager.get_display_size()

        lidar_range = 2.0 * float(self.lidar_sensor.sensor_options['range'])

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

        self.lidar_sensor.data = lidar_img

        if self.client.display_manager.render_enabled():
            self.lidar_sensor.surface = pygame.surfarray.make_surface(lidar_img)

    def gnss_callback(self, data):
        self.gnss_sensor.data = [data.latitude, data.longitude]

    def imu_callback(self, data):
        self.imu_sensor.data = {
            'gyro': data.gyroscope,
            'accel': data.accelerometer,
            'compass': data.compass,
        }

    def lane_inv_callback(self, data):
        pass

    def collision_callback(self, data):
        pass

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()
