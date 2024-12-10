from __future__ import annotations
from src import CarlaClient


def main():
    try:
        client = CarlaClient()
        client.run_vo()
    finally:
        pass


if __name__ == '__main__':
    main()
