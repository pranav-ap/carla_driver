from __future__ import annotations

import asyncio

from CarlaClient import CarlaClient


def main():
    try:
        with CarlaClient() as client:
            client.game_loop()
    finally:
        pass


if __name__ == '__main__':
    main()

# async def main():
#     with CarlaClient() as client:
#         await client.game_loop2()
#
# if __name__ == "__main__":
#     asyncio.run(main())

