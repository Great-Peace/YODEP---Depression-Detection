"""Tests for the C2 F0 feature exclusion assertion.

This is the most critical correctness test in the pipeline.  If the C2
condition accidentally includes any F0-related features, the core scientific
claim of the thesis is compromised.
"""

from __future__ import annotations

import pytest


def test_assert_no_f0_passes_clean_feature_names():
    """C2 assertion passes when no F0-related feature names are present."""
    from src.features.pipeline import assert_no_f0_features

    clean_names = [
        "mfcc_0_mean", "mfcc_0_std", "spectral_centroid_mean",
        "jitter_local_pct", "shimmer_local_pct", "hnr_mean_db", "cpps_mean",
    ]
    # Should not raise
    assert_no_f0_features(clean_names, "C2")


def test_assert_no_f0_raises_on_f0_column():
    """C2 assertion raises AssertionError when an F0 column is present."""
    from src.features.pipeline import assert_no_f0_features

    bad_names = [
        "mfcc_0_mean",
        "F0semitoneFrom27.5Hz_sma3nz_amean",  # F0 column
        "shimmer_local_pct",
    ]
    with pytest.raises(AssertionError) as exc_info:
        assert_no_f0_features(bad_names, "C2")
    assert "F0" in str(exc_info.value) or "C2" in str(exc_info.value)


def test_assert_no_f0_raises_on_pitch_keyword():
    """C2 assertion raises for any 'pitch' substring."""
    from src.features.pipeline import assert_no_f0_features

    bad_names = ["mfcc_0_mean", "pitch_mean", "hnr_mean_db"]
    with pytest.raises(AssertionError):
        assert_no_f0_features(bad_names, "C2")


def test_assert_no_f0_raises_on_fundamental_keyword():
    """C2 assertion raises for 'fundamental' substring."""
    from src.features.pipeline import assert_no_f0_features

    bad_names = ["fundamental_frequency_mean", "shimmer_apq3_pct"]
    with pytest.raises(AssertionError):
        assert_no_f0_features(bad_names, "C2")


def test_assert_no_f0_raises_on_prosodic_keyword():
    """C2 assertion raises for 'prosodic' substring."""
    from src.features.pipeline import assert_no_f0_features

    bad_names = ["prosodic_feature_1", "hnr_mean_db"]
    with pytest.raises(AssertionError):
        assert_no_f0_features(bad_names, "C2")


def test_assert_no_f0_not_enforced_for_c1():
    """Assertion is NOT enforced for C1 (F0 is expected in C1)."""
    from src.features.pipeline import assert_no_f0_features

    names_with_f0 = [
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "mfcc_0_mean",
        "shimmer_local_pct",
    ]
    # Should not raise for C1
    assert_no_f0_features(names_with_f0, "C1")


def test_assert_no_f0_not_enforced_for_c3():
    """Assertion is NOT enforced for C3 (different condition)."""
    from src.features.pipeline import assert_no_f0_features

    names = ["jitter_local_pct", "shimmer_local_pct", "hnr_mean_db", "cpps_mean"]
    # C3 has no F0 features anyway but the assertion doesn't check C3
    assert_no_f0_features(names, "C3")


def test_c2_mm_raises_on_f0():
    """C2_MM (multimodal C2) also triggers the F0 assertion."""
    from src.features.pipeline import assert_no_f0_features

    bad_names = ["pitch_variance", "mfcc_0_mean", "bert_cls_0"]
    with pytest.raises(AssertionError):
        assert_no_f0_features(bad_names, "C2_MM")


def test_c2_xfer_raises_on_f0():
    """C2_XFER (transfer C2) also triggers the F0 assertion."""
    from src.features.pipeline import assert_no_f0_features

    bad_names = ["F0_mean", "shimmer_local_pct"]
    with pytest.raises(AssertionError):
        assert_no_f0_features(bad_names, "C2_XFER")
