"""Glottal source feature extraction via Parselmouth (Python-Praat interface).

Extracts jitter, shimmer, HNR, and CPPS from voiced segments.
Used in conditions C1, C2, and C3.  These features arise from vocal fold
behaviour and are not phonemically regulated in any known language, making
them candidates for language-agnostic depression biomarkers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..data.audio_utils import get_voiced_frames
from ..utils.logger import get_logger

logger = get_logger(__name__)

GLOTTAL_FEATURE_NAMES = [
    "jitter_local_pct",
    "jitter_local_abs_sec",
    "jitter_rap_pct",
    "jitter_ppq5_pct",
    "shimmer_local_pct",
    "shimmer_local_db",
    "shimmer_apq3_pct",
    "shimmer_apq5_pct",
    "hnr_mean_db",
    "hnr_std_db",
    "cpps_mean",
]


def extract_glottal(
    audio: np.ndarray,
    sr: int,
    voiced_only: bool = True,
    min_voiced_duration_seconds: float = 0.5,
) -> Tuple[np.ndarray, List[str]]:
    """Extract glottal source features using Parselmouth.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 audio signal at *sr* Hz.
    sr : int
        Sample rate in Hz.
    voiced_only : bool
        If *True*, extract features on voiced segments only.
    min_voiced_duration_seconds : float
        Minimum voiced duration; falls back to full signal if not met.

    Returns
    -------
    features : np.ndarray, shape (11,)
        Glottal feature vector.
    feature_names : list of str
        Names matching :data:`GLOTTAL_FEATURE_NAMES`.

    Notes
    -----
    Jitter and shimmer are extracted using Praat's standard voice report.
    CPPS is computed via cepstral analysis on voiced frames.
    Features correspond to those used in Williamson et al. (2016) and
    Trevino et al. (2011) for glottal-based depression detection.
    """
    if voiced_only:
        signal = get_voiced_frames(audio, sr, min_voiced_duration_seconds)
    else:
        signal = audio

    features = _extract_voice_report(signal, sr)
    features = np.array(features, dtype=np.float32)

    logger.debug("Extracted %d glottal features.", len(features))
    return features, GLOTTAL_FEATURE_NAMES


def _extract_voice_report(audio: np.ndarray, sr: int) -> List[float]:
    """Run Praat voice report and return the 11 glottal features.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal (voiced segment or full).
    sr : int
        Sample rate.

    Returns
    -------
    list of float
        [jitter_local_pct, jitter_local_abs, jitter_rap, jitter_ppq5,
         shimmer_local_pct, shimmer_local_db, shimmer_apq3, shimmer_apq5,
         hnr_mean, hnr_std, cpps_mean]
    """
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        duration = snd.get_total_duration()

        if duration < 0.1:
            logger.warning("Audio too short for voice report (%.3f s); returning zeros.", duration)
            return [0.0] * 11

        # Point process for jitter/shimmer
        pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)
        point_process = call([snd, pitch], "To PointProcess (cc)")

        # Jitter
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_local_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

        # Shimmer
        shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_local_db = call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # HNR
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        hnr_mean = call(harmonicity, "Get mean", 0, 0)
        hnr_std = call(harmonicity, "Get standard deviation", 0, 0)

        # CPPS
        cpps_mean = _compute_cpps(audio, sr)

        def _safe(v: float) -> float:
            return float(v) if not (np.isnan(v) or np.isinf(v)) else 0.0

        return [
            _safe(jitter_local * 100),
            _safe(jitter_local_abs),
            _safe(jitter_rap * 100),
            _safe(jitter_ppq5 * 100),
            _safe(shimmer_local * 100),
            _safe(shimmer_local_db),
            _safe(shimmer_apq3 * 100),
            _safe(shimmer_apq5 * 100),
            _safe(hnr_mean),
            _safe(hnr_std),
            _safe(cpps_mean),
        ]

    except Exception as exc:
        logger.warning("Glottal extraction failed: %s — returning zeros.", exc)
        return [0.0] * 11


def _compute_cpps(audio: np.ndarray, sr: int) -> float:
    """Compute Cepstral Peak Prominence Smoothed (CPPS) via cepstral analysis.

    Parameters
    ----------
    audio : np.ndarray
        Voiced audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    float
        Mean CPPS across voiced frames in dB.  Returns 0.0 on failure.

    Notes
    -----
    CPPS is a measure of periodicity/voicing quality that correlates with
    breathiness and is robust to noise.  It is computed as the difference
    (in dB) between the cepstral peak value and the regression line value
    at the same quefrency, smoothed over time.

    Hillenbrand & Houde (1996); Heman-Ackah et al. (2003).
    """
    try:
        frame_length = int(sr * 0.025)  # 25 ms
        hop_length = int(sr * 0.010)    # 10 ms
        n_fft = 2 ** int(np.ceil(np.log2(frame_length)))

        # Framing
        frames = []
        for start in range(0, len(audio) - frame_length, hop_length):
            frames.append(audio[start: start + frame_length])

        if not frames:
            return 0.0

        cpps_values = []
        for frame in frames:
            # Windowed power spectrum
            windowed = frame * np.hanning(len(frame))
            spec = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
            spec = np.maximum(spec, 1e-10)
            log_spec = 10 * np.log10(spec)

            # Real cepstrum
            cep = np.fft.irfft(log_spec)
            cep_db = 20 * np.log10(np.maximum(np.abs(cep), 1e-10))

            # Only look in quefrency range corresponding to F0 (75–600 Hz)
            q_min = int(sr / 600)
            q_max = int(sr / 75)
            q_min = max(q_min, 1)
            q_max = min(q_max, len(cep_db) - 1)

            if q_max <= q_min:
                continue

            cep_region = cep_db[q_min:q_max]
            peak_val = np.max(cep_region)

            # Regression line
            x = np.arange(len(cep_region))
            coeffs = np.polyfit(x, cep_region, 1)
            peak_idx = np.argmax(cep_region)
            regression_at_peak = np.polyval(coeffs, peak_idx)

            cpps_values.append(peak_val - regression_at_peak)

        return float(np.mean(cpps_values)) if cpps_values else 0.0

    except Exception as exc:
        logger.warning("CPPS computation failed: %s", exc)
        return 0.0
