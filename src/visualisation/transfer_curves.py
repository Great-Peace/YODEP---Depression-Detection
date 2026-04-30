"""Cross-lingual transfer performance bar chart (Extra 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 300


def plot_transfer_curves(
    transfer_results: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot within-language vs cross-lingual accuracy for C1, C2, C3.

    Parameters
    ----------
    transfer_results : pd.DataFrame
        Rows matching Table 3 schema:
        ``condition``, ``train_language``, ``test_language``,
        ``classifier``, ``f1_macro``.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the saved PNG.

    Notes
    -----
    If C2 and C3 show higher cross-lingual accuracy than C1, that is
    strong evidence for the Acoustic-Linguistic Confound hypothesis (Extra 5).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "transfer_curves_C1_C2_C3.png"

    conditions = transfer_results["condition"].unique()
    n_conditions = len(conditions)

    # Define the four comparison groups
    comparisons = [
        ("EN → EN", "EN", "EN"),
        ("YO → YO", "YO", "YO"),
        ("EN → YO", "EN", "YO"),
        ("YO → EN", "YO", "EN"),
    ]
    x_labels = [c[0] for c in comparisons]
    x = np.arange(len(comparisons))
    width = 0.8 / max(n_conditions, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_conditions))

    for i, cond in enumerate(sorted(conditions)):
        cond_df = transfer_results[transfer_results["condition"] == cond]
        heights = []
        for _, train_lang, test_lang in comparisons:
            match = cond_df[
                (cond_df["train_language"] == train_lang)
                & (cond_df["test_language"] == test_lang)
            ]
            val = float(match["f1_macro"].mean()) if not match.empty else 0.0
            heights.append(val)

        offset = (i - n_conditions / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            heights,
            width=width,
            label=cond,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("F1 Macro", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Within-Language vs Cross-Lingual Transfer Performance",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(title="Condition", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Highlight cross-lingual bars
    for xi in [2, 3]:
        ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.05, color="orange")

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved transfer curve plot: %s", out_path)
    return out_path
