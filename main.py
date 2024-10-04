from __future__ import annotations
from CarlaClient import CarlaClient


def main():
    try:
        with CarlaClient() as client:
            client.game_loop()
    finally:
        pass


if __name__ == '__main__':
    main()
