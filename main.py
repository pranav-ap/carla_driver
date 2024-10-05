from __future__ import annotations
from CarlaClient import CarlaClient


def main():
    try:
        client = CarlaClient()
        client.run()
    finally:
        pass


if __name__ == '__main__':
    main()
