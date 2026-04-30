"""DAIC-WOZ dataset loader: audio segmentation, PHQ-8 labelling, train/dev split.

Used solely for pipeline validation against published baselines.  Primary
dataset is YODEP.

Directory structure expected under ``data/daic_woz/raw/``:
    <participant_id>/
        <participant_id>_AUDIO.wav
        <participant_id>_TRANSCRIPT.csv  (contains start/stop times)
    labels.csv  (columns: participant_id, PHQ_Binary, PHQ8_Score)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

PHQ8_THRESHOLD = 10
LABEL_MAP = {0: "non_depressed", 1: "depressed"}


def load_labels(labels_path: Path, phq8_threshold: int = PHQ8_THRESHOLD) -> pd.DataFrame:
    """Load and binarise PHQ-8 labels.

    Parameters
    ----------
    labels_path : Path
        Path to ``labels.csv`` containing at minimum ``Participant_ID``
        and ``PHQ8_Score`` columns.
    phq8_threshold : int
        Score >= threshold → depressed (label=1).  Default 10 per standard
        DAIC-WOZ protocol.

    Returns
    -------
    pd.DataFrame
        Columns: ``participant_id``, ``phq8_score``, ``label``.
    """
    df = pd.read_csv(labels_path)
    # normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    id_col = next(
        (c for c in df.columns if "participant" in c or "id" in c), None
    )
    score_col = next(
        (c for c in df.columns if "phq8" in c and "score" in c.lower()), None
    )

    if id_col is None or score_col is None:
        raise ValueError(
            f"Cannot find participant_id or phq8_score columns in {labels_path}. "
            f"Found: {list(df.columns)}"
        )

    df = df.rename(columns={id_col: "participant_id", score_col: "phq8_score"})
    df["participant_id"] = df["participant_id"].astype(str).str.zfill(3)
    df["label"] = (df["phq8_score"] >= phq8_threshold).astype(int)

    n_dep = df["label"].sum()
    n_non = len(df) - n_dep
    logger.info(
        "DAIC-WOZ labels: %d depressed, %d non-depressed (threshold=%d).",
        n_dep,
        n_non,
        phq8_threshold,
    )
    return df[["participant_id", "phq8_score", "label"]]


def load_split_manifest(
    raw_dir: Path,
    labels_df: pd.DataFrame,
    split: str = "train",
    phq8_threshold: int = PHQ8_THRESHOLD,
) -> pd.DataFrame:
    """Build a manifest for one DAIC-WOZ split.

    Parameters
    ----------
    raw_dir : Path
        Root directory of the raw DAIC-WOZ data.
    labels_df : pd.DataFrame
        Labels from :func:`load_labels`.
    split : str
        ``"train"`` or ``"dev"``.
    phq8_threshold : int
        Binarisation threshold (passed through for logging).

    Returns
    -------
    pd.DataFrame
        Columns: ``participant_id``, ``audio_path``, ``transcript_path``,
        ``label``, ``phq8_score``.
    """
    raw_dir = Path(raw_dir)
    split_file = raw_dir / f"{split}_split.csv"

    if split_file.exists():
        split_ids = pd.read_csv(split_file)
        id_col = split_ids.columns[0]
        participant_ids = split_ids[id_col].astype(str).str.zfill(3).tolist()
    else:
        # Fallback: use all participants present in raw_dir subdirectories
        logger.warning(
            "%s not found; using all participants in raw_dir.", split_file
        )
        participant_ids = [
            p.name for p in raw_dir.iterdir() if p.is_dir()
        ]

    records: List[Dict] = []
    missing = 0

    for pid in participant_ids:
        audio_path = raw_dir / pid / f"{pid}_AUDIO.wav"
        transcript_path = raw_dir / pid / f"{pid}_TRANSCRIPT.csv"

        if not audio_path.exists():
            logger.warning("Missing audio for participant %s.", pid)
            missing += 1
            continue

        label_row = labels_df[labels_df["participant_id"] == pid]
        if label_row.empty:
            logger.warning("No label found for participant %s; skipping.", pid)
            continue

        records.append(
            {
                "participant_id": pid,
                "audio_path": audio_path,
                "transcript_path": transcript_path if transcript_path.exists() else None,
                "label": int(label_row["label"].iloc[0]),
                "phq8_score": float(label_row["phq8_score"].iloc[0]),
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        "DAIC-WOZ %s split: %d participants loaded, %d missing.",
        split,
        len(df),
        missing,
    )
    return df


def load_daic_woz(
    raw_dir: Path,
    phq8_threshold: int = PHQ8_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load DAIC-WOZ train and dev manifests.

    Parameters
    ----------
    raw_dir : Path
        Root of the DAIC-WOZ raw data directory.
    phq8_threshold : int
        PHQ-8 score >= threshold → depressed label.

    Returns
    -------
    train_df : pd.DataFrame
        Training split manifest.
    dev_df : pd.DataFrame
        Development split manifest.

    Notes
    -----
    The DAIC-WOZ corpus is not publicly available without signing a DUA.
    See https://dcapswoz.ict.usc.edu/ for access.  The pipeline validation
    section of the thesis uses this data only to confirm that our feature
    extraction produces metrics comparable to published results (Williamson
    et al., 2016; Valstar et al., 2016).
    """
    labels_path = raw_dir / "labels.csv"
    if not labels_path.exists():
        # Try common alternative
        labels_path = raw_dir / "phq8_labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Cannot find labels.csv or phq8_labels.csv in {raw_dir}. "
            "Place the DAIC-WOZ label file there."
        )

    labels_df = load_labels(labels_path, phq8_threshold=phq8_threshold)
    train_df = load_split_manifest(raw_dir, labels_df, split="train")
    dev_df = load_split_manifest(raw_dir, labels_df, split="dev")
    return train_df, dev_df
