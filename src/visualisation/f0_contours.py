"""F0 contour visualisation per speaker and summary figure (Extra 1).

Generates publication-quality PNG figures (300 DPI) showing F0 over time
for NORMAL vs DEPRESSED conditions per speaker per language.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.audio_utils import load_and_preprocess
from ..features.prosodic import extract_f0_contour
from ..utils.logger import get_logger

logger = get_logger(__name__)

CONDITION_COLORS = {"NORMAL": "#2196F3", "DEPRESSED": "#F44336"}
CONDITION_ALPHA = {"NORMAL": 0.8, "DEPRESSED": 0.8}
FIGURE_DPI = 300


def plot_f0_contour_speaker(
    manifest_df: pd.DataFrame,
    speaker_id: str,
    language: str,
    audio_cfg: dict,
    output_dir: Path,
) -> Path:
    """Plot F0 contour for one speaker in one language.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        YODEP manifest.
    speaker_id : str
        Speaker to plot.
    language : str
        ``"EN"`` or ``"YO"``.
    audio_cfg : dict
        Audio config section.
    output_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG file.

    Notes
    -----
    NORMAL and DEPRESSED contours are overlaid on the same axes.
    Each sentence is shown as a separate segment separated by grey shading.
    The output filename follows the convention:
    ``f0_contour_{speaker_id}_{language}.png``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"f0_contour_{speaker_id}_{language}.png"

    subset = manifest_df[
        (manifest_df["speaker_id"] == speaker_id)
        & (manifest_df["language"] == language)
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(
        f"F0 Contour — Speaker {speaker_id}, Language {language}",
        fontsize=12,
        fontweight="bold",
    )

    time_offset = 0.0
    for condition in ["NORMAL", "DEPRESSED"]:
        cond_rows = subset[subset["condition"] == condition].sort_values("sentence_id")
        for _, row in cond_rows.iterrows():
            try:
                audio, sr, valid = load_and_preprocess(
                    Path(row["filepath"]),
                    target_sr=audio_cfg["sample_rate"],
                    trim=audio_cfg.get("trim_silence", True),
                    silence_threshold_db=audio_cfg.get("silence_threshold_db", -40.0),
                    min_duration_seconds=0.5,
                )
                if not valid:
                    continue
                times, f0_vals = extract_f0_contour(audio, sr)
                voiced_mask = ~np.isnan(f0_vals)

                if condition == "NORMAL":
                    ax.plot(
                        times[voiced_mask] + time_offset,
                        f0_vals[voiced_mask],
                        color=CONDITION_COLORS[condition],
                        alpha=CONDITION_ALPHA[condition],
                        linewidth=1.5,
                        label=condition if row["sentence_id"] == "S1" else None,
                    )
                else:
                    ax.plot(
                        times[voiced_mask] + time_offset,
                        f0_vals[voiced_mask],
                        color=CONDITION_COLORS[condition],
                        alpha=CONDITION_ALPHA[condition],
                        linewidth=1.5,
                        linestyle="--",
                        label=condition if row["sentence_id"] == "S1" else None,
                    )
            except Exception as exc:
                logger.warning(
                    "F0 contour failed for %s: %s", row["filepath"], exc
                )

        if condition == "NORMAL":
            # Use NORMAL time axis to track sentence boundaries
            cond_rows2 = subset[subset["condition"] == "NORMAL"].sort_values("sentence_id")
            t_off = 0.0
            for _, row2 in cond_rows2.iterrows():
                try:
                    audio2, sr2, valid2 = load_and_preprocess(
                        Path(row2["filepath"]),
                        target_sr=audio_cfg["sample_rate"],
                        trim=False,
                        min_duration_seconds=0.5,
                    )
                    if valid2:
                        dur = len(audio2) / sr2
                        t_off += dur
                except Exception:
                    t_off += 3.0  # fallback 3s
            time_offset = 0.0  # reset for DEPRESSED overlay

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("F0 (Hz)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F0 contour: %s", out_path)
    return out_path


def plot_f0_summary(
    manifest_df: pd.DataFrame,
    audio_cfg: dict,
    output_dir: Path,
) -> Path:
    """Plot mean F0 contour across all speakers per language per condition.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        YODEP manifest.
    audio_cfg : dict
        Audio config section.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the saved summary PNG.

    Notes
    -----
    This is the key figure for the paper.  It visually demonstrates the
    Acoustic-Linguistic Confound or its absence.  Shows mean F0 ± standard
    error as shaded bands.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "f0_contour_summary.png"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    languages = ["EN", "YO"]
    lang_labels = {"EN": "English", "YO": "Yoruba"}

    for ax, lang in zip(axes, languages):
        ax.set_title(f"{lang_labels[lang]}", fontsize=13, fontweight="bold")

        for condition in ["NORMAL", "DEPRESSED"]:
            all_f0: List[np.ndarray] = []
            subset = manifest_df[
                (manifest_df["language"] == lang)
                & (manifest_df["condition"] == condition)
            ]
            for _, row in subset.iterrows():
                try:
                    audio, sr, valid = load_and_preprocess(
                        Path(row["filepath"]),
                        target_sr=audio_cfg["sample_rate"],
                        trim=audio_cfg.get("trim_silence", True),
                        min_duration_seconds=0.5,
                    )
                    if not valid:
                        continue
                    _, f0_vals = extract_f0_contour(audio, sr)
                    voiced = f0_vals[~np.isnan(f0_vals)]
                    if len(voiced) > 0:
                        all_f0.append(voiced)
                except Exception as exc:
                    logger.warning("F0 extract failed: %s", exc)

            if not all_f0:
                continue

            # Interpolate all contours to a common 100-point grid
            grid = np.linspace(0, 1, 100)
            interp_contours = []
            for contour in all_f0:
                x_orig = np.linspace(0, 1, len(contour))
                interp = np.interp(grid, x_orig, contour)
                interp_contours.append(interp)

            interp_arr = np.array(interp_contours)
            mean_f0 = np.mean(interp_arr, axis=0)
            se_f0 = np.std(interp_arr, axis=0) / np.sqrt(len(interp_arr))

            color = CONDITION_COLORS[condition]
            linestyle = "--" if condition == "DEPRESSED" else "-"
            ax.plot(grid, mean_f0, color=color, linewidth=2, linestyle=linestyle, label=condition)
            ax.fill_between(grid, mean_f0 - se_f0, mean_f0 + se_f0, color=color, alpha=0.2)

        ax.set_xlabel("Normalised Time", fontsize=11)
        ax.set_ylabel("F0 (Hz)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Mean F0 Contour: NORMAL vs DEPRESSED across all speakers",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F0 summary figure: %s", out_path)
    return out_path


def plot_all_speakers(
    manifest_df: pd.DataFrame,
    audio_cfg: dict,
    output_dir: Path,
) -> List[Path]:
    """Generate F0 contour plots for every speaker and language combination.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        YODEP manifest.
    audio_cfg : dict
        Audio config section.
    output_dir : Path
        Output directory for PNG files.

    Returns
    -------
    list of Path
        Paths to all generated figures.
    """
    paths = []
    speakers = manifest_df["speaker_id"].unique()
    languages = manifest_df["language"].unique()

    for spk in sorted(speakers):
        for lang in sorted(languages):
            subset = manifest_df[
                (manifest_df["speaker_id"] == spk)
                & (manifest_df["language"] == lang)
            ]
            if subset.empty:
                continue
            p = plot_f0_contour_speaker(manifest_df, spk, lang, audio_cfg, output_dir)
            paths.append(p)

    # Summary figure
    paths.append(plot_f0_summary(manifest_df, audio_cfg, output_dir))
    return paths
