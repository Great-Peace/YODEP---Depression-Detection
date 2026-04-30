"""Prosodic feature extraction via openSMILE eGeMAPS v02.

Extracts F0 statistics, energy statistics, and speech rate.
**These features are ONLY used in condition C1.** They must never appear
in C2 or C3 feature vectors (enforced by the pipeline assertion in
``pipeline.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# eGeMAPS v02 F0 column names (semitone scale from 27.5 Hz)
F0_COLUMNS = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
]

# Additional prosodic columns from eGeMAPS
ENERGY_COLUMNS = [
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
]

SPEECH_RATE_COLUMN = "speech_rate_voiced_ratio"

# All column names that are considered F0/prosodic — used for C2 assertion
F0_RELATED_KEYWORDS = ["F0", "pitch", "fundamental", "prosodic"]


def extract_prosodic(
    audio: np.ndarray,
    sr: int,
    feature_set: str = "eGeMAPSv02",
) -> Tuple[np.ndarray, List[str]]:
    """Extract prosodic features using openSMILE eGeMAPS v02.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 audio signal at *sr* Hz.
    sr : int
        Sample rate (must be 16000 for eGeMAPS).
    feature_set : str
        openSMILE feature set name.  Defaults to ``"eGeMAPSv02"``.

    Returns
    -------
    features : np.ndarray, shape (n_features,)
        Concatenated prosodic feature vector: F0 stats + energy stats +
        speech rate.
    feature_names : list of str
        Human-readable feature names matching positions in *features*.

    Notes
    -----
    openSMILE eGeMAPS v02 extracts 88 features; we select only the
    prosodic subset relevant to depression analysis (Schuller et al., 2016;
    Ringeval et al., 2019).
    """
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # opensmile expects a file or numpy array; we pass the array directly
    result: pd.DataFrame = smile.process_signal(audio, sr)

    # Select F0 columns present in this version
    f0_cols = [c for c in F0_COLUMNS if c in result.columns]
    energy_cols = [c for c in ENERGY_COLUMNS if c in result.columns]

    f0_vals = result[f0_cols].values.flatten() if f0_cols else np.array([])
    energy_vals = result[energy_cols].values.flatten() if energy_cols else np.array([])

    # Speech rate via voiced/unvoiced ratio
    speech_rate = _estimate_speech_rate(audio, sr)

    features = np.concatenate([f0_vals, energy_vals, [speech_rate]])
    feature_names = f0_cols + energy_cols + [SPEECH_RATE_COLUMN]

    logger.debug("Extracted %d prosodic features.", len(features))
    return features.astype(np.float32), feature_names


def _estimate_speech_rate(audio: np.ndarray, sr: int) -> float:
    """Estimate speech rate as voiced frame proportion via Parselmouth.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    float
        Proportion of voiced frames (0–1).  Higher = faster / more voiced.
    """
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)
        frame_times = pitch.ts()
        voiced = sum(
            1 for t in frame_times
            if not np.isnan(pitch.get_value_at_time(t))
        )
        ratio = voiced / max(len(frame_times), 1)
        return float(ratio)
    except Exception as exc:
        logger.warning("Speech rate estimation failed: %s", exc)
        return 0.0


def extract_f0_contour(
    audio: np.ndarray,
    sr: int,
    time_step: float = 0.01,
    f0_min: float = 75.0,
    f0_max: float = 600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frame-by-frame F0 values for visualisation and statistics.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 audio signal.
    sr : int
        Sample rate in Hz.
    time_step : float
        Analysis frame step in seconds.
    f0_min : float
        Minimum F0 in Hz (Praat default: 75 Hz).
    f0_max : float
        Maximum F0 in Hz (Praat default: 600 Hz).

    Returns
    -------
    times : np.ndarray
        Frame centre times in seconds.
    f0_values : np.ndarray
        F0 in Hz for each frame.  NaN for unvoiced frames.

    Notes
    -----
    Used by ``visualisation/f0_contours.py`` and ``evaluation/significance.py``
    for the Wilcoxon signed-rank test (Extra 2).
    """
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = call(snd, "To Pitch", time_step, f0_min, f0_max)

    times = pitch.ts()
    f0_values = np.array([pitch.get_value_at_time(t) for t in times])
    return np.array(times), f0_values
