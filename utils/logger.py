"""Structured logger — consistent formatting across all agents and tools."""

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)

    effective_level = level or "INFO"
    logger.setLevel(getattr(logging, effective_level.upper(), logging.INFO))
    logger.propagate = False

    return logger


def configure_root_logger(level: str = "INFO") -> None:
    """Configure the root logger (called once at application startup)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
    )
