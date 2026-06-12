"""
utils/logger.py
---------------
Centralised logging setup.
Import `get_logger(__name__)` in every module.
"""

import logging
import sys
from config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FMT


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — avoid duplicate handlers
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger
