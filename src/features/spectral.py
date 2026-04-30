"""Spectral feature extraction via librosa.

Extracts MFCCs (+ delta, delta-delta), spectral centroid, spectral rolloff,
and zero-crossing rate.  Used in conditions C1 and C2.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


def extract_spectral(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    include_delta: bool = True,
    include_delta2: bool = True,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, List[str]]:
    """Extract spectral features from an audio signal.

    Computes MFCCs (mean + SD), optionally delta and delta-delta MFCC
    statistics, spectral centroid (mean + SD), spectral rolloff (mean + SD),
    and zero-crossing rate (mean + SD).

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 audio signal.
    sr : int
        Sample rate in Hz.
    n_mfcc : int
        Number of MFCC coefficients.  Defaults to 13.
    include_delta : bool
        If *True*, append delta MFCC statistics.
    include_delta2 : bool
        If *True*, append delta-delta MFCC statistics.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length between frames.

    Returns
    -------
    features : np.ndarray, shape (n_features,)
        Feature vector.  Total length = n_mfcc*2 + (n_mfcc*2 if delta) +
        (n_mfcc*2 if delta2) + 6 (centroid, rolloff, ZCR each mean+SD).
    feature_names : list of str
        Human-readable feature names.

    Notes
    -----
    Total feature count with defaults (n_mfcc=13): 26 + 26 + 26 + 6 = 84.
    All features use mean and standard deviation over time frames.
    """
    import librosa

    feature_parts = []
    names = []

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    feature_parts.extend([mfcc_mean, mfcc_std])
    names += [f"mfcc_{i}_mean" for i in range(n_mfcc)]
    names += [f"mfcc_{i}_std" for i in range(n_mfcc)]

    # --- Delta MFCCs ---
    if include_delta:
        delta = librosa.feature.delta(mfccs)
        feature_parts.extend([np.mean(delta, axis=1), np.std(delta, axis=1)])
        names += [f"mfcc_delta_{i}_mean" for i in range(n_mfcc)]
        names += [f"mfcc_delta_{i}_std" for i in range(n_mfcc)]

    # --- Delta-delta MFCCs ---
    if include_delta2:
        delta2 = librosa.feature.delta(mfccs, order=2)
        feature_parts.extend([np.mean(delta2, axis=1), np.std(delta2, axis=1)])
        names += [f"mfcc_delta2_{i}_mean" for i in range(n_mfcc)]
        names += [f"mfcc_delta2_{i}_std" for i in range(n_mfcc)]

    # --- Spectral centroid ---
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    feature_parts.append(np.array([np.mean(centroid), np.std(centroid)]))
    names += ["spectral_centroid_mean", "spectral_centroid_std"]

    # --- Spectral rolloff ---
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    feature_parts.append(np.array([np.mean(rolloff), np.std(rolloff)]))
    names += ["spectral_rolloff_mean", "spectral_rolloff_std"]

    # --- Zero-crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
    feature_parts.append(np.array([np.mean(zcr), np.std(zcr)]))
    names += ["zcr_mean", "zcr_std"]

    features = np.concatenate(feature_parts).astype(np.float32)
    logger.debug("Extracted %d spectral features.", len(features))
    return features, names
