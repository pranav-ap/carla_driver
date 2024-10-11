from __future__ import annotations
from utils.logger_setup import logger

import random
import carla
import cv2
import numpy as np
import pygame

from abc import ABC, abstractmethod
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from Sensor import RGBCameraSensor
from ClientDisplay import PyGameWidget
from FeatueMatcher import FeatureMatcher
from ObjectDetector import ObjectDetector


class PlayerActions:
    """
    @staticmethod
    def visual_odometry(player, sensor_name: str, screen_name: str, image, frame_counter):
        if frame_counter % 3 == 0 and player.previous_image is not None and (player.feature_matcher_future is None or player.feature_matcher_future.done()):
            sensor: RGBCameraSensor = player.sensors[sensor_name]
            image = sensor.undistort_image(image)

            player.feature_matcher_future = player.executor.submit(
                player.feature_matcher.predict,
                player.previous_image,
                image
            )

        if player.feature_matcher_future and player.feature_matcher_future.done():
            try:
                package = player.feature_matcher_future.result()
                matched_image, kp1, des1, q1, kp2, des2, q2 = package
                
                sensor: RGBCameraSensor = player.sensors[sensor_name]
                T = PlayerActions.get_pose(q1, q2, sensor.K)

                screen = player.screens[screen_name]
                screen.surface = pygame.surfarray.make_surface(matched_image.swapaxes(0, 1))

            except Exception as e:
                print(f"Feature Matching Exception : {e}")

        player.previous_image = image
    """

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
                    screen = player.widgets[screen_name]
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

    def __init__(self):
        if self.__initialized:
            return

        self.__initialized = True

        from CarlaClient import CarlaClient
        self.client: CarlaClient = CarlaClient()

        self.actor = self._init_actor()
        assert self.actor is not None

        self.sensors: Dict[str, RGBCameraSensor] = self._init_sensors()
        self.widgets: Dict[str, PyGameWidget] = self._init_widgets()

    """
    Setup & Cleanup
    """

    def _init_actor(self):
        world = self.client.world

        bp = world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        if bp.has_attribute('color'):
            color = bp.get_attribute('color').recommended_values[0]
            bp.set_attribute('color', color)

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Try to spawn the actor in a free spawn point
        for spawn_point in spawn_points:
            actor = world.try_spawn_actor(bp, spawn_point)
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

        self.executor = ThreadPoolExecutor(max_workers=1)

        self.feature_matcher = FeatureMatcher()
        self.feature_matcher_future = None

        self.current_pose_estimate = None
        self.left_image = None
        self.right_image = None

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
            callback=self.rgb_camera_front_callback
        )

        return sensors

    def _init_widgets(self):
        widgets = dict()

        widgets['VO Screen'] = PyGameWidget(
            grid_start_position=(0, 0),  # row, col
            span=(1, 2)  # row, col
        )

        return widgets

    def cleanup(self):
        super().cleanup()

        from utils import visualize_paths2
        visualize_paths2(
            self.gt_path,
            self.estimated_path,
            "Visual Odometry",
            file_out=f"output/plot.html"
        )

        self.executor.shutdown(wait=False)

    """
    Sensor Callbacks
    """

    def rgb_camera_front_callback(self, image: carla.Image):
        frame_number = image.frame_number
        image = RGBCameraSensor.numpy_from_image(image)
        try:
            self.visual_odometry('RGB Camera Front', image, frame_number)
        except AttributeError as e:
            print(f'VO attribute error again! {e}')

    """
    Act
    """

    def visual_odometry(self, sensor_name: str, image, frame_counter):
        if self.left_image is None:
            image = self.sensors[sensor_name].undistort_image(image)
            self.left_image = image.copy()
            return

        if frame_counter % 2 == 0 and (self.feature_matcher_future is None or self.feature_matcher_future.done()):
            image = self.sensors[sensor_name].undistort_image(image)
            self.right_image = image.copy()

            self.feature_matcher_future = self.executor.submit(
                self.feature_matcher.predict_classical,
                self.left_image,
                image
            )

        if self.right_image is not None and self.feature_matcher_future is not None and self.feature_matcher_future.done():
            self.left_image = self.right_image

            package = self.feature_matcher_future.result()
            matched_image, kp1, des1, q1, kp2, des2, q2, repeatability_score = package

            K = self.sensors[sensor_name].K
            self.visual_odometry_future_handling(q1, q2, K)
            self.visual_odometry_future_ui_handling(matched_image, repeatability_score)

    def visual_odometry_future_handling(self, q1, q2, K):
        # Get the ground truth pose for comparison
        gt_pose: carla.Transform = self.actor.get_transform()
        gt_pose_homo_matrix = gt_pose.get_matrix()
        gt_pose_homo_matrix = np.array(gt_pose_homo_matrix)

        if self.current_pose_estimate is None:
            self.current_pose_estimate = gt_pose_homo_matrix.copy()
        else:
            # Ensure enough matches exist
            if len(q1) < 8 or len(q2) < 8:
                print("Not enough matches to estimate pose.")
                return None

            # relative pose T (between current frame and previous frame)
            T = self.estimate_relative_pose(q1, q2, K)
            # Update current estimated pose
            self.current_pose_estimate = np.matmul(self.current_pose_estimate, np.linalg.inv(T))

        self.gt_path.append((gt_pose.location.x, gt_pose.location.z))
        self.estimated_path.append((self.current_pose_estimate[0, 3], self.current_pose_estimate[2, 3]))

        # Calculate the difference between ground truth and estimated pose
        # translation_error, rotation_error = self.compute_pose_difference(gt_pose, self.current_pose_estimate)
        # print(translation_error, rotation_error)

        # return translation_error, rotation_error

    def visual_odometry_future_ui_handling(self, matched_image, repeatability_score):
        # matched_image = put_text(
        #     matched_image,
        #     org="bottom_left",
        #     text=f"Repeatability : {repeatability_score:.4f}",
        #          # f"Errors        : {translation_error, rotation_error}",
        #     font_scale=0.7,
        # )

        print(f"repeatability_score : {repeatability_score}")

        widget = self.widgets['VO Screen']
        widget.surface = pygame.surfarray.make_surface(matched_image.swapaxes(0, 1))

    @staticmethod
    def compute_pose_difference(gt_transform: carla.Transform, estimated_pose: np.ndarray):
        """
        Computes the translation and rotation difference between ground truth and estimated poses.

        Parameters:
        - gt_transform (carla.Transform): The ground truth transform.
        - estimated_pose (np.ndarray): The estimated pose in a 4x4 pose matrix.

        Returns:
        - translation_error (float): The translation error in the x-z direction.
        - rotation_error (float): The rotation error in degrees.
        """

        # Extract estimated translation from the 4x4 pose matrix (focus on x and z)
        estimated_location = carla.Location(
            x=estimated_pose[0, 3],
            y=estimated_pose[1, 3],
            z=estimated_pose[2, 3]
        )

        # Extract x and z coordinates from both ground truth and estimated pose
        gt_x, gt_z = gt_transform.location.x, gt_transform.location.z
        est_x, est_z = estimated_location.x, estimated_location.z

        # Calculate the distance in the x-z direction (horizontal plane you're interested in)
        translation_error = ((gt_x - est_x) ** 2 + (gt_z - est_z) ** 2) ** 0.5

        """
        estimated_pose = np.array([
            [r11, r12, r13, tx],
            [r21, r22, r23, ty],
            [r31, r32, r33, tz],
            [0,   0,   0,   1]
        ])
        """

        # Extract yaw (heading direction) from the estimated pose's rotation matrix
        estimated_yaw = np.degrees(np.arctan2(estimated_pose[1, 0], estimated_pose[0, 0]))

        # Compute the rotation error (difference in yaw angle)
        rotation_error = abs(gt_transform.rotation.yaw - estimated_yaw)

        # Normalize the rotation error to stay within [-180, 180] degrees
        if rotation_error > 180:
            rotation_error = 360 - rotation_error

            return translation_error, rotation_error

    @staticmethod
    def estimate_relative_pose(q1, q2, K) -> np.ndarray:
        """
        Computes the transformation matrix T
        that represents the relative movement (rotation R and translation t)
        between the two frames using RANSAC filtering.
        """
        E, inliers = cv2.findEssentialMat(q1, q2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, q1, q2, cameraMatrix=K, mask=inliers)

        # Convert translation and rotation to homogeneous transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.squeeze(t)

        return T


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

    def _init_widgets(self):
        screens = dict()
        screens['OD Screen'] = PyGameWidget(None, (0, 0))
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



