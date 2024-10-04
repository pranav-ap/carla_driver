import pygame

from config import config


class DisplayManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DisplayManager, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, grid_size):
        if self.__initialized:
            return

        self.__initialized = True

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("CARLA Driver")

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
            scaled_surface = pygame.transform.scale(s.surface, (w // self.grid_size[1], h // self.grid_size[0]))
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

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render_enabled(self):
        return self.display is not None

    def cleanup(self):
        for s in self.sensor_list:
            s.cleanup()

        pygame.quit()


