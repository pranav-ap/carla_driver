import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_a, K_d, K_s, K_w


class KeyboardController:
    def parse_events(self, player_actor):
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                return True

        return self.control(player_actor)

    def control(self, player_actor):
        keys = pygame.key.get_pressed()
        control = player_actor.get_control()

        self.handle_throttle(control, keys)
        self.handle_steering(control, keys)
        control.hand_brake = keys[K_SPACE]

        player_actor.apply_control(control)
        return False

    @staticmethod
    def handle_throttle(control, keys):
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        else:
            control.throttle = 0

    @staticmethod
    def handle_steering(control, keys):
        if keys[K_a]:
            control.steer = max(-1.0, control.steer - 0.05)
        elif keys[K_d]:
            control.steer = min(1.0, control.steer + 0.05)
        else:
            control.steer *= 0.9  # Smoothly decays the steering angle to 0 when no keys are pressed
