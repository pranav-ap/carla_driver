from __future__ import annotations

import carla
import pygame
import numpy as np
import random
import weakref

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pygame.locals import K_ESCAPE, K_SPACE
from pygame.locals import K_a, K_d, K_s, K_w


@dataclass
class DriverConfig:
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_FOV = 110

    FRAME_RATE = 20


config = DriverConfig()


class DisplayManager:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.window_size = self.init_window_size()
        self.display = pygame.display.set_mode(self.window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_list = []

    def init_window_size(self):
        original_image_width, original_image_height = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
        print(f'Original Image Dim : {original_image_width, original_image_height}')

        # Primary monitor
        max_window_width = pygame.display.get_desktop_sizes()[0][0]

        # Calculate the new width for each image
        total_images = self.grid_size[0] * self.grid_size[1]  # Total images = rows * columns
        max_image_width = max_window_width // self.grid_size[1]  # Width for each column
        new_image_width = min(original_image_width, max_image_width)

        # Ensure height scaling proportionally
        scale_factor = new_image_width / original_image_width
        new_image_height = int(original_image_height * scale_factor)
        print(f'New Image Dim : {new_image_width, new_image_height}')

        window_size = (new_image_width * self.grid_size[1], new_image_height * self.grid_size[0])  # Adjusted for two rows

        return window_size

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            if s.surface is None or s.display_pos is None:
                continue

            offset = self.get_display_offset(s.display_pos)
            # Scale the surface
            w, h = self.get_window_size()
            scaled_surface = pygame.transform.scale(s.surface, (w // self.grid_size[1], h // self.grid_size[0]))  # Adjusted for scaling
            self.display.blit(scaled_surface, offset)

        pygame.display.flip()

    def get_display_offset(self, display_pos):
        # Calculate the offset based on the grid position
        row, col = display_pos
        new_image_width = self.window_size[0] // self.grid_size[1]
        new_image_height = self.window_size[1] // self.grid_size[0]

        return col * new_image_width, row * new_image_height

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [
            int(self.window_size[0] / self.grid_size[1]),
            int(self.window_size[1] / self.grid_size[0])
        ]

    # def get_display_offset(self, display_pos):
    #     dis_size = self.get_display_size()
    #     return [int(display_pos[1] * dis_size[0]), int(display_pos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render_enabled(self):
        return self.display is not None

    def cleanup(self):
        for s in self.sensor_list:
            s.cleanup()


class SensorDevice:
    def __init__(self, client_weak_self, sensor_type, spawn_point, sensor_options, display_pos):
        self.client_weak_self = client_weak_self

        self.sensor = self.init_sensor(
            sensor_type,
            spawn_point,
            sensor_options
        )

        self.sensor_options = sensor_options
        self.data = self.init_sensor_data(sensor_type)

        self.display_pos: Tuple[int] | None = display_pos
        self.surface = None

        client: CarlaClient = client_weak_self()
        client.display_manager.add_sensor(self)

    @staticmethod
    def init_sensor_data(sensor_type):
        if sensor_type == 'sensor.camera.rgb':
            return np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 4))
        elif sensor_type == 'sensor.lidar.ray_cast':
            return np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 4))
        elif sensor_type == 'gnss':
            return [0, 0]
        elif sensor_type == 'imu':
            return {
                'gyro': carla.Vector3D(),
                'accel': carla.Vector3D(),
                'compass': 0
            }

    def init_sensor(self, sensor_type, spawn_point, sensor_options):
        client: CarlaClient = self.client_weak_self()
        sensor_bp = client.world.get_blueprint_library().find(sensor_type)

        sensor = None

        if sensor_type == 'sensor.camera.rgb':
            sensor_bp.set_attribute('image_size_x', f'{config.IMAGE_WIDTH}')
            sensor_bp.set_attribute('image_size_y', f'{config.IMAGE_HEIGHT}')
            sensor_bp.set_attribute('fov', f'{config.IMAGE_FOV}')

            sensor = client.world.spawn_actor(sensor_bp, spawn_point, attach_to=client.car.actor)
            sensor.listen(self.rgb_callback)

        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_bp.set_attribute('range', '100')
            sensor_bp.set_attribute('dropoff_general_rate', sensor_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            sensor_bp.set_attribute('dropoff_intensity_limit', sensor_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            sensor_bp.set_attribute('dropoff_zero_intensity', sensor_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            sensor = client.world.spawn_actor(sensor_bp, spawn_point, attach_to=client.car.actor)
            sensor.listen(self.lidar_callback)

        if sensor is None:
            return None

        for key in sensor_options:
            sensor_bp.set_attribute(key, sensor_options[key])

        return sensor

    """
    Sensor Callbacks
    """

    def rgb_callback(self, image):
        client: CarlaClient = self.client_weak_self()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if client.display_manager.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def lidar_callback(self, image):
        client: CarlaClient = self.client_weak_self()

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

        if client.display_manager.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

    def cleanup(self):
        if self.sensor is not None:
            self.sensor.destroy()


class EgoCar:
    def __init__(self):
        self.rgb_camera_front: SensorDevice | None = None
        self.rgb_camera_left: SensorDevice | None = None
        self.rgb_camera_right: SensorDevice | None = None
        self.rgb_camera_rear: SensorDevice | None = None
        self.lidar_sensor: SensorDevice | None = None

        self.actor = None

    def setup(self, client_weak_self):
        self._setup_actor(client_weak_self)
        self._setup_sensors(client_weak_self)

    def _setup_actor(self, client_weak_self):
        client: CarlaClient = client_weak_self()
        bp = client.world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        spawn_point = random.choice(client.world.get_map().get_spawn_points())
        self.actor = client.world.spawn_actor(bp, spawn_point)

    def _setup_sensors(self, client_weak_self):
        self.rgb_camera_front = SensorDevice(
            client_weak_self,
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=1.5, z=2.1), carla.Rotation(yaw=0)),
            sensor_options={},
            display_pos=[0, 1]
        )

        self.rgb_camera_left = SensorDevice(
            client_weak_self,
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=0, y=-1.0, z=2.1), carla.Rotation(yaw=-90)),
            sensor_options={},
            display_pos=[0, 0]
        )

        self.rgb_camera_right = SensorDevice(
            client_weak_self,
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=0, y=1.0, z=2.1), carla.Rotation(yaw=90)),
            sensor_options={},
            display_pos=[0, 2]
        )

        self.rgb_camera_rear = SensorDevice(
            client_weak_self,
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=-1.5, z=2.1), carla.Rotation(yaw=180)),
            sensor_options={},
            display_pos=[1, 1]
        )

        # self.lidar_sensor = SensorDevice(
        #     client_weak_self,
        #     'sensor.lidar.ray_cast',
        #     carla.Transform(carla.Location(z=2.5), carla.Rotation(pitch=-15, yaw=90)),
        #     sensor_options={
        #         'channels': '64',
        #         'range': '100',
        #         'points_per_second': '56000',
        #         'rotation_frequency': '20'
        #     },
        #     display_pos=[1, 2]
        # )

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()


class CarlaClient:
    def __init__(self):
        self.client = None
        self.world = None
        self.car = EgoCar()
        self.display_manager: DisplayManager | None = None

    """
    Manual Control
    """

    def control(self):
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = self.car.actor.get_control()
        self.handle_throttle(control, keys)
        self.handle_steering(control, keys)
        control.hand_brake = keys[K_SPACE]

        self.car.actor.apply_control(control)
        return False

    @staticmethod
    def handle_throttle(control, keys):
        control.throttle = 1 if keys[K_w] or keys[K_s] else 0
        control.reverse = keys[K_s]

    @staticmethod
    def handle_steering(control, keys):
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0

    """
    Game Loop
    """

    def __enter__(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("CARLA Driver")

        """
        Client
        """

        self.client = carla.Client('127.0.0.1', 2000)
        assert self.client is not None
        self.client.set_timeout(10.0)

        """
        World
        """

        self.world = self.client.get_world()
        assert self.world is not None

        self.set_synchronous_mode(True)

        """
        Display Manager
        """

        self.display_manager = DisplayManager(
            grid_size=[2, 3],  # rows, cols
        )

        """
        Car
        """

        client_weak_self = weakref.ref(self)
        self.car.setup(client_weak_self)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.car.cleanup()

        if self.display_manager is not None:
            self.display_manager.cleanup()

        self.set_synchronous_mode(False)
        pygame.quit()

    def update_spectator_location(self):
        spectator = self.world.get_spectator()

        distance_behind = 5
        vehicle_transform = self.car.actor.get_transform()
        offset = vehicle_transform.get_forward_vector() * -distance_behind

        # noinspection PyArgumentList
        spectator.set_transform(
            carla.Transform(
                # Adjust z for the height above ground
                vehicle_transform.location + offset + carla.Location(z=2.5),
                # Look forward in the direction of the car
                carla.Rotation(pitch=0, yaw=vehicle_transform.rotation.yaw)
            )
        )

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def game_loop(self):
        pygame_clock = pygame.time.Clock()

        while True:
            self.world.tick()
            self.update_spectator_location()
            pygame_clock.tick_busy_loop(config.FRAME_RATE)

            self.display_manager.render()

            pygame.event.pump()

            # Check if the window is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    return

            if self.control():
                return


def main():
    try:
        with CarlaClient() as client:
            client.game_loop()
    finally:
        pass


if __name__ == '__main__':
    main()
