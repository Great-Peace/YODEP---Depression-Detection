"""Tests for feature extraction modules using synthetic audio."""

from __future__ import annotations

import numpy as np
import pytest


def _make_sine(freq=220, sr=16000, duration=2.0):
    """Generate a pure sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32), sr


def test_spectral_feature_shape():
    from src.features.spectral import extract_spectral

    audio, sr = _make_sine()
    features, names = extract_spectral(audio, sr, n_mfcc=13)
    assert features.ndim == 1
    assert len(features) == len(names)
    # 13*2 (mfcc) + 13*2 (delta) + 13*2 (delta2) + 6 = 84
    assert len(features) == 84


def test_spectral_no_delta():
    from src.features.spectral import extract_spectral

    audio, sr = _make_sine()
    features, names = extract_spectral(
        audio, sr, n_mfcc=13, include_delta=False, include_delta2=False
    )
    # 13*2 (mfcc mean+std) + 6 = 32
    assert len(features) == 32


def test_spectral_no_nan():
    from src.features.spectral import extract_spectral

    audio, sr = _make_sine()
    features, _ = extract_spectral(audio, sr)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_spectral_feature_names_match_vector():
    from src.features.spectral import extract_spectral

    audio, sr = _make_sine()
    features, names = extract_spectral(audio, sr, n_mfcc=13)
    assert len(features) == len(names)


def test_glottal_feature_count():
    from src.features.glottal import extract_glottal, GLOTTAL_FEATURE_NAMES

    audio, sr = _make_sine(freq=150, duration=2.0)
    features, names = extract_glottal(audio, sr, voiced_only=False)
    assert len(features) == 11
    assert len(names) == 11
    assert names == GLOTTAL_FEATURE_NAMES


def test_glottal_returns_float32():
    from src.features.glottal import extract_glottal

    audio, sr = _make_sine(freq=150, duration=2.0)
    features, _ = extract_glottal(audio, sr, voiced_only=False)
    assert features.dtype == np.float32


def test_cache_path_deterministic():
    from pathlib import Path
    from src.utils.cache import cache_path

    p1 = cache_path(Path(".cache"), Path("P01_EN_NORMAL_S1_T1.wav"), "C1", {"n_mfcc": 13})
    p2 = cache_path(Path(".cache"), Path("P01_EN_NORMAL_S1_T1.wav"), "C1", {"n_mfcc": 13})
    assert p1 == p2


def test_cache_invalidated_on_config_change():
    from pathlib import Path
    from src.utils.cache import cache_path

    p1 = cache_path(Path(".cache"), Path("P01_EN_NORMAL_S1_T1.wav"), "C1", {"n_mfcc": 13})
    p2 = cache_path(Path(".cache"), Path("P01_EN_NORMAL_S1_T1.wav"), "C1", {"n_mfcc": 26})
    assert p1 != p2


def test_audio_utils_load_and_preprocess_missing_file():
    from pathlib import Path
    from src.data.audio_utils import load_and_preprocess

    with pytest.raises(FileNotFoundError):
        load_and_preprocess(Path("nonexistent_file.wav"))


def test_audio_utils_duration_check_pass():
    from src.data.audio_utils import check_duration

    audio = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds
    assert check_duration(audio, sr=16000, min_duration_seconds=1.5) is True


def test_audio_utils_duration_check_fail():
    from src.data.audio_utils import check_duration

    audio = np.zeros(16000, dtype=np.float32)  # 1 second
    assert check_duration(audio, sr=16000, min_duration_seconds=1.5) is False
