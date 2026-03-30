"""Shared logging configuration.

A single call to ``get_logger(__name__)`` in every module guarantees:
  - Consistent format across all ELIQSIR output.
  - Log level driven by ``settings.log_level`` (env var ``LOG_LEVEL``).
  - No duplicate handlers even when the function is called multiple times.

Usage::

    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Starting extraction...")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# We lazily resolve the log level from settings to avoid a circular import
# (config.py imports nothing from src.utils, so the direction is safe).
_root_configured = False


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger with a guaranteed stream handler.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Override the log level for *this specific logger* (e.g. ``"DEBUG"``).
        When ``None`` the project-wide level from ``settings.log_level`` is used.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    global _root_configured

    # Configure the root logger exactly once.
    if not _root_configured:
        # Import here to break any potential circular dependency at module
        # load time; config only uses std-lib, so this is always safe.
        try:
            from src.config import settings  # noqa: PLC0415

            root_level = settings.log_level
        except Exception:
            root_level = "INFO"

        logging.basicConfig(
            level=root_level,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            stream=sys.stdout,
        )
        _root_configured = True

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level.upper())

    # Guard against duplicate handlers when modules are reloaded
    # (e.g. importlib.reload() in notebooks).  The root logger already
    # owns the stream handler, so child loggers should never have their
    # own handlers — remove any that crept in during a reload.
    if logger.handlers:
        logger.handlers.clear()

    logger.propagate = True

    return logger
