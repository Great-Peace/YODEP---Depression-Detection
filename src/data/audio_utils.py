"""Audio preprocessing utilities: resampling, silence trimming, VAD, duration check.

All functions operate on file paths and return NumPy arrays or metadata dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_audio(
    path: Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to *target_sr*.

    Parameters
    ----------
    path : Path
        Path to the ``.wav`` (or any soundfile-supported format) file.
    target_sr : int
        Target sample rate in Hz.  Defaults to 16 000 Hz.
    mono : bool
        If *True*, downmix multi-channel audio to mono.

    Returns
    -------
    audio : np.ndarray, shape (n_samples,)
        Normalised float32 audio signal.
    sr : int
        Sample rate (always *target_sr*).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim == 2 and mono:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Remove leading and trailing silence from an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    sr : int
        Sample rate (unused currently, reserved for future frame-time calc).
    threshold_db : float
        Energy threshold below which frames are considered silent.
    frame_length : int
        Number of samples per frame for energy computation.
    hop_length : int
        Hop size between frames.

    Returns
    -------
    np.ndarray
        Trimmed audio signal.  Returns the original if trimming would leave
        fewer than 0.5 s of audio.
    """
    import librosa

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=abs(threshold_db),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if len(trimmed) < sr * 0.5:
        logger.warning("Trimmed audio is very short; returning original.")
        return audio
    return trimmed


def check_duration(
    audio: np.ndarray,
    sr: int,
    min_duration_seconds: float = 1.5,
    filename: str = "",
) -> bool:
    """Check that an audio signal meets the minimum duration requirement.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    min_duration_seconds : float
        Minimum acceptable duration in seconds.
    filename : str
        Filename for log messages (informational only).

    Returns
    -------
    bool
        *True* if duration >= *min_duration_seconds*, *False* otherwise.
    """
    duration = len(audio) / sr
    if duration < min_duration_seconds:
        logger.warning(
            "File '%s' is %.2f s — below minimum %.2f s; will be skipped.",
            filename,
            duration,
            min_duration_seconds,
        )
        return False
    return True


def get_voiced_frames(
    audio: np.ndarray,
    sr: int,
    min_voiced_duration_seconds: float = 0.5,
) -> np.ndarray:
    """Extract voiced portions of an audio signal using Parselmouth VAD.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (float32, mono).
    sr : int
        Sample rate in Hz.
    min_voiced_duration_seconds : float
        Minimum total voiced duration required.  Returns *None* if not met.

    Returns
    -------
    np.ndarray
        Concatenated voiced samples.  Returns full signal if insufficient
        voiced frames are detected (fallback).

    Notes
    -----
    Uses Parselmouth's voiced/unvoiced segmentation via autocorrelation-based
    pitch analysis (Praat algorithm).
    """
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)
        voiced_flag = np.array(
            [pitch.get_value_at_time(t) for t in pitch.ts()]
        )
        is_voiced = ~np.isnan(voiced_flag)

        if is_voiced.sum() == 0:
            logger.warning("No voiced frames found; using full signal as fallback.")
            return audio

        # Map voiced pitch frame times back to audio samples
        hop = len(audio) / len(is_voiced)
        voiced_samples = []
        for i, voiced in enumerate(is_voiced):
            if voiced:
                start = int(i * hop)
                end = int(min((i + 1) * hop, len(audio)))
                voiced_samples.append(audio[start:end])

        if not voiced_samples:
            return audio

        voiced_audio = np.concatenate(voiced_samples)
        total_duration = len(voiced_audio) / sr

        if total_duration < min_voiced_duration_seconds:
            logger.warning(
                "Total voiced duration %.2f s < min %.2f s; using full signal.",
                total_duration,
                min_voiced_duration_seconds,
            )
            return audio

        return voiced_audio

    except Exception as exc:
        logger.warning("VAD failed (%s); using full signal as fallback.", exc)
        return audio


def load_and_preprocess(
    path: Path,
    target_sr: int = 16000,
    trim: bool = True,
    silence_threshold_db: float = -40.0,
    min_duration_seconds: float = 1.5,
) -> Tuple[np.ndarray, int, bool]:
    """Load, resample, trim, and validate a single audio file.

    Parameters
    ----------
    path : Path
        Path to the audio file.
    target_sr : int
        Target sample rate.
    trim : bool
        Whether to apply silence trimming.
    silence_threshold_db : float
        Threshold for silence trimming.
    min_duration_seconds : float
        Minimum valid duration after trimming.

    Returns
    -------
    audio : np.ndarray
        Preprocessed audio signal.
    sr : int
        Sample rate.
    valid : bool
        *False* if the file failed duration check and should be skipped.
    """
    audio, sr = load_audio(path, target_sr=target_sr)

    if trim:
        audio = trim_silence(audio, sr, threshold_db=silence_threshold_db)

    valid = check_duration(audio, sr, min_duration_seconds, filename=path.name)
    return audio, sr, valid
