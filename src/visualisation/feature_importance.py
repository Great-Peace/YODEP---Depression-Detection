"""Random Forest feature importance bar chart (Extra 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 300


def plot_feature_importance(
    importance_df: pd.DataFrame,
    condition: str,
    language: str,
    output_dir: Path,
    top_n: int = 20,
) -> Path:
    """Plot top-N feature importances as a horizontal bar chart.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of :func:`~models.random_forest.get_feature_importances`.
        Columns: ``feature_name``, ``importance``.
    condition : str
        Condition identifier (e.g. ``"C1"``).
    language : str
        Language filter label for the title.
    output_dir : Path
        Output directory.
    top_n : int
        Number of features to display.

    Returns
    -------
    Path
        Path to the saved PNG.

    Notes
    -----
    If F0 features dominate in English but not Yoruba, that is direct
    evidence for the Acoustic-Linguistic Confound hypothesis (Extra 4).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"feature_importance_{condition}_{language}_RF.png"

    top = importance_df.head(top_n).copy()

    # Colour-code by feature family
    def _get_color(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["f0", "pitch", "fundamental", "prosodic", "speech_rate"]):
            return "#F44336"  # red — F0/prosodic
        elif any(k in n for k in ["mfcc", "spectral", "zcr", "rolloff", "centroid"]):
            return "#2196F3"  # blue — spectral
        elif any(k in n for k in ["jitter", "shimmer", "hnr", "cpps"]):
            return "#4CAF50"  # green — glottal
        elif any(k in n for k in ["bert", "hubert", "wavlm", "emb"]):
            return "#FF9800"  # orange — embeddings
        return "#9E9E9E"  # grey — other

    colors = [_get_color(n) for n in top["feature_name"]]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(
        range(len(top)),
        top["importance"].values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature_name"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances — {condition} ({language})",
        fontsize=12,
        fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F44336", label="F0 / Prosodic"),
        Patch(facecolor="#2196F3", label="Spectral / MFCC"),
        Patch(facecolor="#4CAF50", label="Glottal (jitter/shimmer/HNR/CPPS)"),
        Patch(facecolor="#FF9800", label="SSL / Text Embeddings"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot: %s", out_path)
    return out_path
