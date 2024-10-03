import carla
import pygame
import numpy as np
import random
import weakref

from dataclasses import dataclass
from typing import Dict, Any, List
from pygame.locals import K_ESCAPE, K_SPACE
from pygame.locals import K_a, K_d, K_s, K_w


@dataclass
class DriverConfig:
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_FOV = 110

    FRAME_RATE = 20


config = DriverConfig()


class SensorData:
    def __init__(self):
        self.rgb_image: np.array = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 4))
        self.depth_image: np.array = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 4))

        self.gnss: List[int] = [0, 0]

        # noinspection PyArgumentList
        self.imu: Dict[str, Any] = {
            'gyro': carla.Vector3D(),
            'accel': carla.Vector3D(),
            'compass': 0
        }


class SensorDevices:
    def __init__(self):
        self.rgb_camera = None
        self.depth_camera = None
        self.gnss_sensor = None
        self.imu_sensor = None

    # noinspection PyArgumentList
    def setup(self, client_weak_self):
        client: CarlaClient = client_weak_self()
        bp_lib = client.world.get_blueprint_library()

        self.rgb_camera = self.setup_sensor(
            client_weak_self,
            bp_lib,
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=1.5, z=2.1))
        )

        self.depth_camera = self.setup_sensor(
            client_weak_self,
            bp_lib,
            'sensor.camera.depth',
            carla.Transform(carla.Location(x=1.5, z=2.1))
        )

        self.gnss_sensor = self.setup_sensor(
            client_weak_self,
            bp_lib,
            'sensor.other.gnss',
            carla.Transform()
        )

        self.imu_sensor = self.setup_sensor(
            client_weak_self,
            bp_lib,
            'sensor.other.imu',
            carla.Transform()
        )

    @staticmethod
    def setup_sensor(client_weak_self, bp_lib, sensor_type, spawn_point):
        client: CarlaClient = client_weak_self()
        sensor_bp = bp_lib.find(sensor_type)
        sensor = client.world.spawn_actor(sensor_bp, spawn_point, attach_to=client.car.actor)

        if sensor_type == 'qqsensor.camera.rgb':
            sensor.set_attribute('image_size_x', f'{config.IMAGE_WIDTH}')
            sensor.set_attribute('image_size_y', f'{config.IMAGE_HEIGHT}')
            sensor.set_attribute('fov', f'{config.IMAGE_FOV}')

        return sensor

    def cleanup(self):
        for sensor in [self.rgb_camera, self.depth_camera, self.gnss_sensor, self.imu_sensor]:
            if sensor is not None:
                sensor.destroy()


class EgoCar:
    def __init__(self):
        self.sensor_devices = SensorDevices()
        self.sensor_data = SensorData()
        self.actor = None

    def setup(self, client_weak_self):
        client: CarlaClient = client_weak_self()
        bp = client.world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        """
        Setup Car Actor 
        """

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        spawn_point = random.choice(client.world.get_map().get_spawn_points())
        self.actor = client.world.spawn_actor(bp, spawn_point)

        """
        Setup Sensors
        """

        self.sensor_devices.setup(client_weak_self)
        self.sensor_devices.rgb_camera.listen(lambda image: self.rgb_callback(client_weak_self, image))
        self.sensor_devices.depth_camera.listen(lambda image: self.depth_callback(image))
        self.sensor_devices.gnss_sensor.listen(lambda data: self.gnss_callback(data))
        self.sensor_devices.imu_sensor.listen(lambda data: self.imu_callback(data))

    def cleanup(self):
        self.sensor_devices.cleanup()

        if self.actor is not None:
            self.actor.destroy()

    """
    Sensor Callbacks
    """

    def rgb_callback(self, client_weak_self, image):
        client: CarlaClient = client_weak_self()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.sensor_data.rgb_image = array

        if client.capture:
            client.image = array
            client.capture = False

    def depth_callback(self, image):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        self.sensor_data.depth_image = np.reshape(image.raw_data, (image.height, image.width, 4))

    def gnss_callback(self, data):
        self.sensor_data.gnss = [data.latitude, data.longitude]

    def imu_callback(self, data):
        self.sensor_data.imu = {
            'gyro': data.gyroscope,
            'accel': data.accelerometer,
            'compass': data.compass
        }


class CarlaClient:
    def __init__(self):
        self.client = None
        self.world = None
        self.car = EgoCar()

        self.display = None
        self.image = None
        self.capture = True

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
    Render
    """

    def draw_rectangle(self):
        rect_color = (255, 0, 0)  # Red color (RGB)
        rect_size = (200, 100)    # Width and height of the rectangle
        rect_thickness = 2        # Border thickness

        screen_center_x = config.IMAGE_WIDTH // 2
        screen_center_y = config.IMAGE_HEIGHT // 2

        # rect_position = (50, 50)  # Top-left corner of the rectangle (x, y)

        rect_position = (
            screen_center_x - rect_size[0] // 2,  # x position
            screen_center_y - rect_size[1] // 2   # y position
        )

        pygame.draw.rect(
            self.display,
            rect_color,
            (*rect_position, *rect_size),
            rect_thickness
        )

    def render(self):
        if self.image is None:
            return

        surface = pygame.surfarray.make_surface(
            self.car.sensor_data.rgb_image.swapaxes(0, 1)
        )

        self.display.blit(surface, (0, 0))
        self.draw_rectangle()

    """
    Game Loop
    """

    def __enter__(self):
        self.initialize_game()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def initialize_game(self):
        pygame.init()

        """
        PyGame Window Settings  
        """

        pygame.display.set_caption("CARLA Driver")

        self.display = pygame.display.set_mode(
            (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        assert self.display is not None

        """
        Client
        """

        self.client = carla.Client('127.0.0.1', 2000)
        assert self.client is not None
        self.client.set_timeout(10.0)

        """
        World & Car
        """

        # noinspection PyArgumentList
        self.world = self.client.get_world()
        assert self.world is not None

        self.set_synchronous_mode(True)

        client_weak_self = weakref.ref(self)
        self.car.setup(client_weak_self)

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

    def run_tick(self):
        self.world.tick()
        self.update_spectator_location()
        self.capture = True

    def cleanup(self):
        self.set_synchronous_mode(False)
        self.car.cleanup()
        pygame.quit()

        print('Cleaned Up!')

    def game_loop(self):
        pygame_clock = pygame.time.Clock()

        while True:
            self.run_tick()
            pygame_clock.tick_busy_loop(config.FRAME_RATE)

            self.render()
            pygame.display.update()
            pygame.event.pump()

            if self.control():
                return


def main():
    with CarlaClient() as client:
        client.game_loop()


if __name__ == '__main__':
    main()
