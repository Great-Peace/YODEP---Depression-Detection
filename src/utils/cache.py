"""Feature caching with SHA-256 config-hash-based invalidation.

Features are stored as ``.npy`` arrays under a structured cache directory.
Cache keys combine the file path, feature condition, and a hash of the
relevant config section so that changing hyperparameters automatically
invalidates stale entries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


def _hash_config(config_dict: dict) -> str:
    """Compute a short SHA-256 hex digest of a serialised config dict.

    Parameters
    ----------
    config_dict : dict
        Any JSON-serialisable dict (typically a features sub-section of
        ``config.yaml``).

    Returns
    -------
    str
        First 12 hex characters of the SHA-256 digest.
    """
    serialised = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]


def cache_path(
    cache_dir: Path,
    audio_path: Path,
    condition: str,
    config_dict: dict,
) -> Path:
    """Compute the ``.npy`` cache file path for a given recording and condition.

    Parameters
    ----------
    cache_dir : Path
        Root cache directory (e.g. ``.cache/features``).
    audio_path : Path
        Absolute path to the source ``.wav`` file.
    condition : str
        Feature condition identifier (e.g. ``"C1"``, ``"C2"``).
    config_dict : dict
        Config sub-section whose hash determines cache validity.

    Returns
    -------
    Path
        Path to the ``.npy`` file (may or may not exist).
    """
    config_hash = _hash_config(config_dict)
    stem = audio_path.stem
    filename = f"{stem}__{condition}__{config_hash}.npy"
    return cache_dir / condition / filename


def load_from_cache(path: Path) -> Optional[np.ndarray]:
    """Load a feature array from cache if it exists.

    Parameters
    ----------
    path : Path
        Cache file path returned by :func:`cache_path`.

    Returns
    -------
    np.ndarray or None
        The cached array, or *None* if the file does not exist.
    """
    if path.exists():
        logger.debug("Cache hit: %s", path.name)
        return np.load(path, allow_pickle=False)
    return None


def save_to_cache(path: Path, array: np.ndarray) -> None:
    """Save a feature array to cache.

    Parameters
    ----------
    path : Path
        Target cache file path.
    array : np.ndarray
        Feature array to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    logger.debug("Cached features to: %s", path.name)


class FeatureCache:
    """High-level cache manager with load/save helpers and hit-rate reporting.

    Parameters
    ----------
    cache_dir : Path
        Root directory for all cached features.
    config_dict : dict
        Config sub-section whose hash invalidates stale entries.

    Examples
    --------
    >>> cache = FeatureCache(Path(".cache/features"), cfg["features"]["glottal"])
    >>> feat = cache.get("P01_EN_NORMAL_S1_T1.wav", "C3")
    >>> if feat is None:
    ...     feat = extract_glottal(audio)
    ...     cache.put("P01_EN_NORMAL_S1_T1.wav", "C3", feat)
    """

    def __init__(self, cache_dir: Path, config_dict: dict) -> None:
        self._dir = Path(cache_dir)
        self._config = config_dict
        self._hits = 0
        self._misses = 0

    def _key(self, audio_path: Path, condition: str) -> Path:
        return cache_path(self._dir, audio_path, condition, self._config)

    def get(self, audio_path: Path, condition: str) -> Optional[np.ndarray]:
        """Return cached array or None.

        Parameters
        ----------
        audio_path : Path
            Source audio file path used as part of the cache key.
        condition : str
            Feature condition identifier.

        Returns
        -------
        np.ndarray or None
        """
        arr = load_from_cache(self._key(audio_path, condition))
        if arr is not None:
            self._hits += 1
        else:
            self._misses += 1
        return arr

    def put(self, audio_path: Path, condition: str, array: np.ndarray) -> None:
        """Save array to cache.

        Parameters
        ----------
        audio_path : Path
            Source audio file path.
        condition : str
            Feature condition identifier.
        array : np.ndarray
            Feature array to store.
        """
        save_to_cache(self._key(audio_path, condition), array)

    def report(self) -> None:
        """Log a summary of cache hit/miss statistics."""
        total = self._hits + self._misses
        logger.info(
            "Feature cache: loaded %d from cache, extracted %d new (total=%d).",
            self._hits,
            self._misses,
            total,
        )
