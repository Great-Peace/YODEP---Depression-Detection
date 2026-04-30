"""YODEP dataset loader: filename parsing, metadata loading, LOSO fold generation.

File naming convention (strict):
    ``[SpeakerID]_[Language]_[Condition]_[Sentence]_[Take].wav``
    e.g. ``P01_EN_NORMAL_S1_T1.wav``, ``P07_YO_DEPRESSED_S3_T2.wav``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Regex for the strict file naming convention
_FNAME_RE = re.compile(
    r"^(?P<speaker_id>P\d{2})_"
    r"(?P<language>EN|YO)_"
    r"(?P<condition>NORMAL|DEPRESSED)_"
    r"(?P<sentence_id>S[1-5])_"
    r"(?P<take>T\d+)\.wav$",
    re.IGNORECASE,
)

LABEL_MAP = {"NORMAL": 0, "DEPRESSED": 1}


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse a YODEP filename into its constituent fields.

    Parameters
    ----------
    filename : str
        Basename of the file (e.g. ``"P01_EN_NORMAL_S1_T1.wav"``).

    Returns
    -------
    dict or None
        Keys: ``speaker_id``, ``language``, ``condition``, ``sentence_id``,
        ``take``.  Returns *None* if the filename does not match the pattern.

    Examples
    --------
    >>> parse_filename("P01_EN_NORMAL_S1_T1.wav")
    {'speaker_id': 'P01', 'language': 'EN', 'condition': 'NORMAL',
     'sentence_id': 'S1', 'take': 'T1'}
    """
    m = _FNAME_RE.match(Path(filename).name)
    if m is None:
        return None
    d = m.groupdict()
    d["language"] = d["language"].upper()
    d["condition"] = d["condition"].upper()
    return d


def load_yodep_manifest(
    raw_dir: Path,
    metadata_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a manifest DataFrame from all valid YODEP wav files.

    Parameters
    ----------
    raw_dir : Path
        Directory containing ``.wav`` files (non-recursive).
    metadata_csv : Path, optional
        Path to ``metadata.csv`` with speaker-level information.  If provided,
        columns from metadata are merged in on ``speaker_id``.

    Returns
    -------
    pd.DataFrame
        One row per recording.  Columns: ``filepath``, ``speaker_id``,
        ``language``, ``condition``, ``sentence_id``, ``take``, ``label``,
        plus metadata columns if supplied.

    Notes
    -----
    Missing files are logged as warnings and skipped.  At the end a summary
    line is printed: "Loaded X speakers, Y recordings."
    """
    raw_dir = Path(raw_dir)
    records: List[Dict] = []
    missing: List[str] = []

    wav_files = sorted(raw_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No .wav files found in %s", raw_dir)

    for wav_path in wav_files:
        parsed = parse_filename(wav_path.name)
        if parsed is None:
            logger.warning("Skipping file with unexpected name: %s", wav_path.name)
            continue
        record = {
            "filepath": wav_path,
            **parsed,
            "label": LABEL_MAP[parsed["condition"]],
        }
        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        logger.warning("No valid YODEP recordings loaded from %s.", raw_dir)
        return df

    # Merge metadata if provided
    if metadata_csv is not None and Path(metadata_csv).exists():
        meta = pd.read_csv(metadata_csv)
        if "speaker_id" in meta.columns:
            df = df.merge(meta, on="speaker_id", how="left")
            logger.info("Merged metadata for %d speakers.", meta["speaker_id"].nunique())
        else:
            logger.warning("metadata.csv missing 'speaker_id' column; skipping merge.")

    n_speakers = df["speaker_id"].nunique()
    n_recordings = len(df)
    logger.info(
        "Loaded %d speakers, %d recordings. Missing: %d files.",
        n_speakers,
        n_recordings,
        len(missing),
    )
    return df


def loso_folds(
    df: pd.DataFrame,
    language: Optional[str] = None,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, str], None, None]:
    """Generate Leave-One-Speaker-Out (LOSO) train/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full manifest DataFrame from :func:`load_yodep_manifest`.
    language : str, optional
        If provided (``"EN"`` or ``"YO"``), restrict to that language only.

    Yields
    ------
    train_df : pd.DataFrame
        All recordings except the held-out speaker.
    test_df : pd.DataFrame
        Recordings for the single held-out speaker.
    held_out_speaker : str
        Speaker ID (e.g. ``"P01"``).

    Notes
    -----
    The LOSO protocol matches the evaluation described in the thesis:
    train on N-1 speakers, test on the remaining speaker.  All metrics
    are averaged over folds post-hoc.
    """
    if language is not None:
        df = df[df["language"] == language.upper()].copy()

    speakers = sorted(df["speaker_id"].unique())
    logger.info(
        "LOSO: %d folds for language=%s.", len(speakers), language or "ALL"
    )

    for spk in speakers:
        test_df = df[df["speaker_id"] == spk].copy()
        train_df = df[df["speaker_id"] != spk].copy()
        logger.debug(
            "Fold speaker=%s — train=%d, test=%d.", spk, len(train_df), len(test_df)
        )
        yield train_df, test_df, spk


def get_text_for_recording(
    sentence_id: str,
    language: str,
    sentences: Dict[str, Dict[str, str]],
) -> str:
    """Look up the fixed sentence text for a recording.

    Parameters
    ----------
    sentence_id : str
        Sentence identifier (e.g. ``"S1"``).
    language : str
        ``"EN"`` or ``"YO"``.
    sentences : dict
        Loaded ``sentences.json`` dict.

    Returns
    -------
    str
        The sentence text.

    Raises
    ------
    KeyError
        If the sentence_id or language is not found.
    """
    return sentences[language.upper()][sentence_id]
