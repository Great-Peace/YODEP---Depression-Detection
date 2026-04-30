"""Global random seed management for full reproducibility.

Sets seeds for Python's built-in :mod:`random`, :mod:`numpy`, :mod:`torch`,
and ``scikit-learn`` (which inherits from NumPy's global state).
"""

from __future__ import annotations

import random

import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The integer seed value. Defaults to 42 (matches ``config.yaml``).

    Notes
    -----
    scikit-learn uses NumPy's global RandomState, so seeding NumPy is
    sufficient.  For reproducible GridSearchCV results always pass
    ``random_state=seed`` explicitly when constructing estimators.

    PyTorch CUDA determinism requires additional environment variable::

        CUBLAS_WORKSPACE_CONFIG=:4096:8

    Set this before launching the script for fully deterministic GPU runs.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch seeds set (seed=%d).", seed)
    except ImportError:
        logger.debug("PyTorch not installed; skipping torch seed.")

    logger.info("All random seeds set to %d.", seed)


def get_device() -> "torch.device":  # noqa: F821
    """Detect and return the best available compute device.

    Returns
    -------
    torch.device
        ``cuda`` if a CUDA GPU is available, ``mps`` on Apple Silicon,
        otherwise ``cpu``.

    Notes
    -----
    Prints a human-readable message at INFO level so every experiment log
    records which device was used.
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)
    return device
