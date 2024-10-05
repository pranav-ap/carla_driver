from __future__ import annotations

import pygame

from pygame.locals import K_ESCAPE, K_SPACE
from pygame.locals import K_a, K_d, K_s, K_w


class KeyboardController:
    def __init__(self, player):
        self.player = player

    def parse_events(self):
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                return True

        return self.control()

    @staticmethod
    def parse_quit_events():
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                return True

        return False

    def control(self):
        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:
            return True

        control = self.player.actor.get_control()

        self.handle_throttle(control, keys)
        self.handle_steering(control, keys)
        control.hand_brake = keys[K_SPACE]

        self.player.actor.apply_control(control)

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

        control.steer = 0

