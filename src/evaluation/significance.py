"""Statistical significance testing for F0 differences between conditions.

Implements Wilcoxon signed-rank test comparing NORMAL vs DEPRESSED F0
distributions per speaker per language.  Reports W statistic, p-value,
and rank-biserial correlation effect size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..features.prosodic import extract_f0_contour
from ..data.audio_utils import load_and_preprocess
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _rank_biserial_correlation(W: float, n: int) -> float:
    """Compute rank-biserial correlation from Wilcoxon W statistic.

    Parameters
    ----------
    W : float
        Wilcoxon signed-rank W statistic.
    n : int
        Number of non-zero differences.

    Returns
    -------
    float
        Effect size r_rb in [-1, 1].
    """
    max_W = n * (n + 1) / 2
    return float(4 * W / (n * (n + 1)) - 1) if max_W > 0 else 0.0


def test_f0_difference_per_speaker(
    manifest_df: pd.DataFrame,
    audio_cfg: dict,
    speaker_id: Optional[str] = None,
    language: Optional[str] = None,
) -> pd.DataFrame:
    """Run Wilcoxon signed-rank test on F0 distributions per speaker.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        YODEP manifest with ``speaker_id``, ``language``, ``condition``,
        ``filepath`` columns.
    audio_cfg : dict
        Audio config section.
    speaker_id : str, optional
        If given, restrict to a single speaker.
    language : str, optional
        If given, restrict to a single language.

    Returns
    -------
    pd.DataFrame
        One row per (speaker, language).  Columns: ``speaker_id``,
        ``language``, ``n_normal_frames``, ``n_depressed_frames``,
        ``f0_mean_normal_hz``, ``f0_mean_depressed_hz``,
        ``f0_separation_hz``, ``wilcoxon_W``, ``p_value``,
        ``effect_size``, ``significant_005``.

    Notes
    -----
    Paired Wilcoxon test compares the distribution of voiced F0 values
    in NORMAL vs DEPRESSED recordings for the same speaker and language.
    Uses all sentences combined.

    Reference: Wilcoxon (1945); Newsom (2020) for rank-biserial correlation.
    """
    df = manifest_df.copy()
    if speaker_id is not None:
        df = df[df["speaker_id"] == speaker_id]
    if language is not None:
        df = df[df["language"] == language.upper()]

    results = []

    for (spk, lang), group in df.groupby(["speaker_id", "language"]):
        normal_rows = group[group["condition"] == "NORMAL"]
        dep_rows = group[group["condition"] == "DEPRESSED"]

        if normal_rows.empty or dep_rows.empty:
            logger.warning("Skipping %s/%s — missing condition data.", spk, lang)
            continue

        normal_f0 = _collect_f0_values(normal_rows, audio_cfg)
        depressed_f0 = _collect_f0_values(dep_rows, audio_cfg)

        if len(normal_f0) == 0 or len(depressed_f0) == 0:
            logger.warning("No voiced F0 frames for %s/%s.", spk, lang)
            continue

        # Wilcoxon requires equal-length paired samples; use min length
        min_len = min(len(normal_f0), len(depressed_f0))
        rng = np.random.RandomState(42)
        n_idx = rng.choice(len(normal_f0), size=min_len, replace=False)
        d_idx = rng.choice(len(depressed_f0), size=min_len, replace=False)

        try:
            stat, p_value = stats.wilcoxon(
                normal_f0[n_idx], depressed_f0[d_idx], alternative="greater"
            )
        except Exception as exc:
            logger.warning("Wilcoxon failed for %s/%s: %s", spk, lang, exc)
            stat, p_value = float("nan"), float("nan")

        effect_size = _rank_biserial_correlation(stat, min_len)

        results.append(
            {
                "speaker_id": spk,
                "language": lang,
                "n_normal_frames": len(normal_f0),
                "n_depressed_frames": len(depressed_f0),
                "f0_mean_normal_hz": float(np.mean(normal_f0)),
                "f0_mean_depressed_hz": float(np.mean(depressed_f0)),
                "f0_separation_hz": float(np.mean(normal_f0) - np.mean(depressed_f0)),
                "wilcoxon_W": float(stat),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "significant_005": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            }
        )

    return pd.DataFrame(results)


def _collect_f0_values(rows: pd.DataFrame, audio_cfg: dict) -> np.ndarray:
    """Collect voiced F0 frames across multiple recordings."""
    all_f0 = []
    for _, row in rows.iterrows():
        try:
            audio, sr, valid = load_and_preprocess(
                Path(row["filepath"]),
                target_sr=audio_cfg["sample_rate"],
                trim=audio_cfg.get("trim_silence", True),
                silence_threshold_db=audio_cfg.get("silence_threshold_db", -40.0),
                min_duration_seconds=audio_cfg.get("min_duration_seconds", 1.5),
            )
            if not valid:
                continue
            _, f0_vals = extract_f0_contour(audio, sr)
            voiced = f0_vals[~np.isnan(f0_vals)]
            all_f0.extend(voiced.tolist())
        except Exception as exc:
            logger.warning("F0 extraction failed for %s: %s", row["filepath"], exc)
    return np.array(all_f0)


def summarise_significance(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Wilcoxon results across speakers per language.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`test_f0_difference_per_speaker`.

    Returns
    -------
    pd.DataFrame
        One row per language with columns matching Table 5 requirements:
        ``language``, ``n_speakers_significant``, ``n_speakers_not_significant``,
        ``median_f0_separation_hz``.
    """
    rows = []
    for lang, group in results_df.groupby("language"):
        n_sig = int(group["significant_005"].sum())
        n_not = int((~group["significant_005"]).sum())
        median_sep = float(group["f0_separation_hz"].median())
        rows.append(
            {
                "language": lang,
                "n_speakers_significant": n_sig,
                "n_speakers_not_significant": n_not,
                "median_f0_separation_hz": median_sep,
            }
        )
    return pd.DataFrame(rows)
