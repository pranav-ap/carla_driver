from __future__ import annotations

import random
import carla

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

    def setup(self):
        self._setup_actor()
        self._setup_sensors()

    def _setup_actor(self):
        from CarlaClient import CarlaClient
        client: CarlaClient = CarlaClient()
        bp = client.world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        spawn_point = random.choice(client.world.get_map().get_spawn_points())
        self.actor = client.world.spawn_actor(bp, spawn_point)

    def _setup_sensors(self):
        self.rgb_camera_front = SensorDevice(
            'sensor.camera.rgb',
            carla.Transform(carla.Location(x=1.5, z=2.1), carla.Rotation(yaw=0)),
            sensor_options={},
            display_pos=[0, 0]
        )

        # self.rgb_camera_rear = SensorDevice(
        #     'sensor.camera.rgb',
        #     carla.Transform(carla.Location(x=-1.5, z=2.1), carla.Rotation(yaw=180)),
        #     sensor_options={},
        #     display_pos=None  # None [0, 1]
        # )
        #
        # self.rgb_camera_left = SensorDevice(
        #     'sensor.camera.rgb',
        #     carla.Transform(carla.Location(x=0, y=-1.0, z=2.1), carla.Rotation(yaw=-90)),
        #     sensor_options={},
        #     display_pos=None  # None [0, 0]
        # )
        #
        # self.rgb_camera_right = SensorDevice(
        #     'sensor.camera.rgb',
        #     carla.Transform(carla.Location(x=0, y=1.0, z=2.1), carla.Rotation(yaw=90)),
        #     sensor_options={},
        #     display_pos=None  # None [0, 2]
        # )

        # self.lidar_sensor = SensorDevice(
        #     'sensor.lidar.ray_cast',
        #     carla.Transform(carla.Location(z=2.5), carla.Rotation(pitch=-15, yaw=90)),
        #     sensor_options={
        #         'channels': '64',
        #         'range': '100',
        #         'points_per_second': '56000',
        #         'rotation_frequency': '10'
        #     },
        #     display_pos=[0, 1]
        # )

    def cleanup(self):
        if self.actor is not None:
            self.actor.destroy()
