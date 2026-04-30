"""Actor quality analysis (Extra 3): F0 separation, LOSO F1, Pearson correlation.

Produces the ranked actor quality table described in the thesis:
speaker_id, language, self_rated_acting_score, f0_mean_normal,
f0_mean_depressed, f0_separation, wilcoxon_p_value, loso_f1_for_speaker.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_actor_quality_table(
    significance_df: pd.DataFrame,
    loso_results: Dict,
    metadata_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the ranked actor quality table.

    Parameters
    ----------
    significance_df : pd.DataFrame
        Output of :func:`~evaluation.significance.test_f0_difference_per_speaker`.
        Must contain: ``speaker_id``, ``language``, ``f0_mean_normal_hz``,
        ``f0_mean_depressed_hz``, ``f0_separation_hz``, ``p_value``.
    loso_results : dict
        Output of :func:`~evaluation.loso.run_loso`.  Uses
        ``fold_metrics`` list to extract per-speaker F1.
    metadata_df : pd.DataFrame, optional
        Speaker metadata with ``speaker_id`` and ``self_rated_acting_score``.

    Returns
    -------
    pd.DataFrame
        Ranked by ``f0_separation_hz`` descending.  Columns match Table 4.
    """
    # Extract per-speaker F1 from LOSO fold metrics
    spk_f1: Dict[str, float] = {}
    for fm in loso_results.get("fold_metrics", []):
        sid = fm.get("speaker_id")
        if sid is not None:
            spk_f1[sid] = fm.get("f1_macro", float("nan"))

    table = significance_df.copy()

    # Add LOSO F1
    table["loso_f1"] = table["speaker_id"].map(spk_f1)

    # Rename columns to match required output
    table = table.rename(
        columns={
            "f0_mean_normal_hz": "f0_mean_normal",
            "f0_mean_depressed_hz": "f0_mean_depressed",
            "f0_separation_hz": "f0_separation_hz",
            "p_value": "wilcoxon_p_value",
        }
    )

    # Merge acting score from metadata
    if metadata_df is not None and "self_rated_acting_score" in metadata_df.columns:
        meta_sub = metadata_df[["speaker_id", "self_rated_acting_score"]].drop_duplicates()
        table = table.merge(meta_sub, on="speaker_id", how="left")
    else:
        table["self_rated_acting_score"] = float("nan")

    # Reorder columns to match Table 4
    col_order = [
        "speaker_id", "language", "self_rated_acting_score",
        "f0_mean_normal", "f0_mean_depressed", "f0_separation_hz",
        "wilcoxon_p_value", "effect_size", "loso_f1",
    ]
    for c in col_order:
        if c not in table.columns:
            table[c] = float("nan")

    table = (
        table[col_order]
        .sort_values("f0_separation_hz", ascending=False)
        .reset_index(drop=True)
    )
    return table


def compute_pearson_f0_vs_f1(actor_table: pd.DataFrame) -> Dict[str, float]:
    """Compute Pearson correlation between F0 separation and LOSO F1.

    Parameters
    ----------
    actor_table : pd.DataFrame
        Output of :func:`build_actor_quality_table`.

    Returns
    -------
    dict
        Keys: ``pearson_r``, ``p_value``, ``n_speakers``.

    Notes
    -----
    Tests PM's "good actor vs bad actor" hypothesis: speakers with larger
    F0 separation should be easier to classify.
    """
    valid = actor_table.dropna(subset=["f0_separation_hz", "loso_f1"])
    if len(valid) < 3:
        logger.warning("Too few observations for Pearson correlation (%d).", len(valid))
        return {"pearson_r": float("nan"), "p_value": float("nan"), "n_speakers": len(valid)}

    r, p = stats.pearsonr(valid["f0_separation_hz"], valid["loso_f1"])
    logger.info(
        "Pearson r(F0 separation, LOSO F1) = %.4f (p=%.4f, n=%d).",
        r, p, len(valid)
    )
    return {"pearson_r": float(r), "p_value": float(p), "n_speakers": len(valid)}
