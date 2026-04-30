"""Structured logging to console and file with colour support.

Provides a factory function that returns a configured :class:`logging.Logger`
instance.  All loggers are children of the root ``yodep`` logger so their
output is controlled from a single place.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog  # optional, degrades gracefully
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False

_ROOT_LOGGER_NAME = "yodep"
_FILE_HANDLER: Optional[logging.FileHandler] = None


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_filename: str = "run.log",
) -> None:
    """Configure the root ``yodep`` logger.

    Parameters
    ----------
    log_dir : Path, optional
        Directory where the log file is written.  If *None*, file logging is
        disabled.
    level : int
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    log_filename : str
        Name of the log file inside *log_dir*.

    Notes
    -----
    Call this once at the start of every experiment script before calling
    :func:`get_logger`.
    """
    global _FILE_HANDLER

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    if root.handlers:
        root.handlers.clear()

    # Console handler
    if _HAS_COLOR:
        fmt = (
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s "
            "%(blue)s%(name)s%(reset)s — %(message)s"
        )
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                fmt,
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename
        _FILE_HANDLER = logging.FileHandler(log_path, encoding="utf-8")
        _FILE_HANDLER.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(_FILE_HANDLER)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``yodep`` namespace.

    Parameters
    ----------
    name : str
        Module name, typically ``__name__``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Feature extraction started.")
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
