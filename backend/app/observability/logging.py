from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stdout, level=level, format="{time} {level} {message}", serialize=True)
