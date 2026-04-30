"""Tests for data loading modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_parse_filename_valid():
    from src.data.yodep_loader import parse_filename

    result = parse_filename("P01_EN_NORMAL_S1_T1.wav")
    assert result is not None
    assert result["speaker_id"] == "P01"
    assert result["language"] == "EN"
    assert result["condition"] == "NORMAL"
    assert result["sentence_id"] == "S1"
    assert result["take"] == "T1"


def test_parse_filename_depressed():
    from src.data.yodep_loader import parse_filename

    result = parse_filename("P07_YO_DEPRESSED_S3_T2.wav")
    assert result is not None
    assert result["speaker_id"] == "P07"
    assert result["language"] == "YO"
    assert result["condition"] == "DEPRESSED"


def test_parse_filename_invalid():
    from src.data.yodep_loader import parse_filename

    assert parse_filename("random_file.wav") is None
    assert parse_filename("P01_EN_NORMAL_S1.wav") is None  # missing take
    assert parse_filename("") is None


def test_parse_filename_case_insensitive():
    from src.data.yodep_loader import parse_filename

    result = parse_filename("P01_en_normal_s1_t1.wav")
    assert result is not None
    assert result["language"] == "EN"
    assert result["condition"] == "NORMAL"


def test_load_yodep_manifest_empty_dir():
    from src.data.yodep_loader import load_yodep_manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        df = load_yodep_manifest(Path(tmpdir))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_loso_folds_basic():
    from src.data.yodep_loader import loso_folds

    # Minimal manifest with 3 speakers, 2 recordings each
    rows = []
    for spk in ["P01", "P02", "P03"]:
        for cond, label in [("NORMAL", 0), ("DEPRESSED", 1)]:
            rows.append({
                "speaker_id": spk, "language": "EN",
                "condition": cond, "label": label,
                "filepath": f"{spk}_EN_{cond}_S1_T1.wav",
            })
    df = pd.DataFrame(rows)

    folds = list(loso_folds(df, language="EN"))
    assert len(folds) == 3
    for train, test, spk in folds:
        assert spk in ["P01", "P02", "P03"]
        assert spk not in train["speaker_id"].values
        assert spk in test["speaker_id"].values


def test_label_map():
    from src.data.yodep_loader import LABEL_MAP

    assert LABEL_MAP["NORMAL"] == 0
    assert LABEL_MAP["DEPRESSED"] == 1
