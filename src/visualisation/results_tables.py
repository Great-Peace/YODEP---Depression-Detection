"""Auto-format result tables as LaTeX and CSV.

Every table is saved as both ``.csv`` and ``.tex`` (LaTeX ``tabular``
environment ready to paste into the paper).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str = "{:.4f}",
) -> str:
    """Convert a DataFrame to a LaTeX table string.

    Parameters
    ----------
    df : pd.DataFrame
        Data to format.
    caption : str
        LaTeX table caption.
    label : str
        LaTeX label for ``\\ref{}``.
    float_format : str
        Python format string for floats.

    Returns
    -------
    str
        Complete ``\\begin{table}...\\end{table}`` block.
    """
    col_fmt = "l" + "r" * (len(df.columns) - 1)

    body = df.to_latex(
        index=False,
        float_format=lambda x: float_format.format(x) if isinstance(x, float) else x,
        column_format=col_fmt,
        escape=True,
        longtable=False,
    )

    latex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )
    return latex


def save_table(
    df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    caption: str = "",
    label: str = "",
) -> None:
    """Save a DataFrame as both CSV and LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        Table data.
    output_dir : Path
        Directory to write files.
    stem : str
        Base filename without extension (e.g. ``"table1_daic_validation"``).
    caption : str
        LaTeX caption.
    label : str
        LaTeX label.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"

    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV table: %s", csv_path)

    if caption:
        latex_str = _df_to_latex(df, caption=caption, label=label or stem)
        tex_path.write_text(latex_str, encoding="utf-8")
        logger.info("Saved LaTeX table: %s", tex_path)


def build_table1_daic(results_list: list) -> pd.DataFrame:
    """Build Table 1: DAIC-WOZ validation results.

    Parameters
    ----------
    results_list : list of dict
        Each dict has keys: ``condition``, ``classifier``, ``f1_macro``,
        ``precision``, ``recall``, ``accuracy``, ``auc``,
        ``ci_lower_95``, ``ci_upper_95``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(results_list, columns=[
        "Condition", "Classifier", "F1_macro", "Precision", "Recall",
        "Accuracy", "AUC", "CI_lower_95", "CI_upper_95"
    ])


def build_table2_yodep_main(results_list: list) -> pd.DataFrame:
    """Build Table 2: YODEP main results (all conditions, all classifiers).

    Parameters
    ----------
    results_list : list of dict
        Keys: ``condition``, ``modality``, ``language``, ``classifier``,
        ``f1_macro_mean``, ``f1_macro_std``, ``precision_mean``,
        ``recall_mean``, ``accuracy_mean``, ``auc_mean``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(results_list, columns=[
        "Condition", "Modality", "Language", "Classifier",
        "F1_macro_mean", "F1_macro_std", "Precision_mean",
        "Recall_mean", "Accuracy_mean", "AUC_mean"
    ])


def build_table3_transfer(results_list: list) -> pd.DataFrame:
    """Build Table 3: Cross-lingual transfer results.

    Parameters
    ----------
    results_list : list of dict
        Keys: ``condition``, ``train_language``, ``test_language``,
        ``classifier``, ``f1_macro``, ``delta_vs_within_language``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(results_list, columns=[
        "Condition", "Train_language", "Test_language",
        "Classifier", "F1_macro", "Delta_vs_within_language"
    ])


def build_table4_actor_quality(actor_df: pd.DataFrame) -> pd.DataFrame:
    """Build Table 4: Actor quality table from actor quality DataFrame.

    Parameters
    ----------
    actor_df : pd.DataFrame
        Output of :func:`~evaluation.actor_quality.build_actor_quality_table`.

    Returns
    -------
    pd.DataFrame
    """
    col_map = {
        "speaker_id": "Speaker_ID",
        "language": "Language",
        "self_rated_acting_score": "Self_rated_acting_score",
        "f0_mean_normal": "F0_mean_normal",
        "f0_mean_depressed": "F0_mean_depressed",
        "f0_separation_hz": "F0_separation_hz",
        "wilcoxon_p_value": "Wilcoxon_p",
        "effect_size": "Effect_size",
        "loso_f1": "LOSO_F1",
    }
    df = actor_df.rename(columns=col_map)
    available = [c for c in col_map.values() if c in df.columns]
    return df[available]


def build_table5_wilcoxon_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build Table 5: Wilcoxon significance summary.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`~evaluation.significance.summarise_significance`,
        optionally extended with group comparison p-value.

    Returns
    -------
    pd.DataFrame
    """
    col_map = {
        "language": "Language",
        "n_speakers_significant": "N_speakers_significant",
        "n_speakers_not_significant": "N_speakers_not_significant",
        "median_f0_separation_hz": "Median_F0_separation_hz",
    }
    df = summary_df.rename(columns=col_map)
    available = [c for c in col_map.values() if c in df.columns]
    return df[available]
