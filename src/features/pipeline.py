"""Feature pipeline: assembles feature vectors per condition.

Enforces the C2 hard constraint: zero F0-related features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.audio_utils import load_and_preprocess
from ..utils.cache import FeatureCache
from ..utils.logger import get_logger
from .glottal import extract_glottal
from .prosodic import extract_prosodic
from .spectral import extract_spectral
from .ssl_embeddings import extract_ssl_embedding
from .text_features import extract_text_embedding

logger = get_logger(__name__)

# Keywords whose presence in any C2 feature name raises AssertionError
F0_FORBIDDEN_KEYWORDS = ["F0", "pitch", "fundamental", "prosodic", "f0"]


def assert_no_f0_features(feature_names: List[str], condition: str) -> None:
    """Assert that a C2 feature vector contains no F0-related features.

    Parameters
    ----------
    feature_names : list of str
        Names of features in the vector.
    condition : str
        Condition identifier (e.g. ``"C2"``).  Only enforced for C2/C2-MM/C2-XFER.

    Raises
    ------
    AssertionError
        If any feature name contains a forbidden keyword.
    """
    if not any(c.startswith("C2") for c in [condition]):
        return

    violations = [
        name for name in feature_names
        if any(kw.lower() in name.lower() for kw in F0_FORBIDDEN_KEYWORDS)
    ]
    assert not violations, (
        f"C2 CONSTRAINT VIOLATED: Feature vector for condition '{condition}' "
        f"contains F0-related features: {violations}. "
        "Remove all prosodic/F0 features before training on C2."
    )


def build_feature_vector(
    audio_path: Path,
    condition: str,
    audio_cfg: dict,
    features_cfg: dict,
    models_cfg: dict,
    cache: Optional[FeatureCache] = None,
    cache_dir: Optional[Path] = None,
    text_info: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble the feature vector for a single recording under a given condition.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file.
    condition : str
        One of: ``"C1"``, ``"C2"``, ``"C3"``, ``"C4"``, ``"C4_WavLM"``,
        ``"C1_MM"``, ``"C2_MM"``, ``"C3_MM"``, ``"C4_MM"``.
    audio_cfg : dict
        Audio section of ``config.yaml``.
    features_cfg : dict
        Features section of ``config.yaml``.
    models_cfg : dict
        Models section of ``config.yaml``.
    cache : FeatureCache, optional
        Cache manager for audio features.
    cache_dir : Path, optional
        Root cache directory for SSL and text embeddings.
    text_info : dict, optional
        Keys: ``text``, ``language``, ``sentence_id``.  Required for
        multimodal conditions (``*_MM``).

    Returns
    -------
    features : np.ndarray
        Feature vector.
    feature_names : list of str
        Names for each dimension.

    Raises
    ------
    AssertionError
        If C2 constraint is violated.
    """
    is_multimodal = condition.endswith("_MM")
    base_condition = condition.replace("_MM", "").replace("_XFER", "")

    # Try cache
    if cache is not None:
        cached = cache.get(audio_path, base_condition)
        if cached is not None:
            audio_features = cached
            audio_names = _get_expected_names(base_condition, features_cfg)
        else:
            audio_features, audio_names = _extract_audio_features(
                audio_path, base_condition, audio_cfg, features_cfg, models_cfg, cache_dir
            )
            if cache is not None:
                cache.put(audio_path, base_condition, audio_features)
    else:
        audio_features, audio_names = _extract_audio_features(
            audio_path, base_condition, audio_cfg, features_cfg, models_cfg, cache_dir
        )

    # C2 hard assertion
    assert_no_f0_features(audio_names, condition)

    if not is_multimodal:
        return audio_features, audio_names

    # Multimodal: append BERT text embedding
    if text_info is None:
        raise ValueError(
            f"text_info is required for multimodal condition {condition}."
        )
    text_emb = extract_text_embedding(
        text=text_info["text"],
        language=text_info["language"],
        sentence_id=text_info["sentence_id"],
        model_name=models_cfg.get("bert", "bert-base-uncased"),
        cache_dir=cache_dir,
    )
    text_names = [f"bert_cls_{i}" for i in range(len(text_emb))]

    features = np.concatenate([audio_features, text_emb])
    names = audio_names + text_names
    return features, names


def _extract_audio_features(
    audio_path: Path,
    base_condition: str,
    audio_cfg: dict,
    features_cfg: dict,
    models_cfg: dict,
    cache_dir: Optional[Path],
) -> Tuple[np.ndarray, List[str]]:
    """Internal dispatcher: extract audio features for a base condition."""
    audio, sr, valid = load_and_preprocess(
        audio_path,
        target_sr=audio_cfg["sample_rate"],
        trim=audio_cfg.get("trim_silence", True),
        silence_threshold_db=audio_cfg.get("silence_threshold_db", -40.0),
        min_duration_seconds=audio_cfg.get("min_duration_seconds", 1.5),
    )
    if not valid:
        logger.warning("Invalid audio: %s — using zero vector.", audio_path.name)
        return _zero_vector(base_condition, features_cfg), _get_expected_names(base_condition, features_cfg)

    parts: List[np.ndarray] = []
    names: List[str] = []

    if base_condition == "C1":
        prosodic, pnames = extract_prosodic(
            audio, sr, features_cfg["prosodic"]["opensmile_feature_set"]
        )
        spectral, snames = extract_spectral(
            audio, sr,
            n_mfcc=features_cfg["spectral"]["n_mfcc"],
            include_delta=features_cfg["spectral"]["include_delta"],
            include_delta2=features_cfg["spectral"]["include_delta2"],
        )
        glottal, gnames = extract_glottal(
            audio, sr,
            voiced_only=features_cfg["glottal"]["voiced_only"],
            min_voiced_duration_seconds=features_cfg["glottal"]["min_voiced_duration_seconds"],
        )
        parts = [prosodic, spectral, glottal]
        names = pnames + snames + gnames

    elif base_condition == "C2":
        spectral, snames = extract_spectral(
            audio, sr,
            n_mfcc=features_cfg["spectral"]["n_mfcc"],
            include_delta=features_cfg["spectral"]["include_delta"],
            include_delta2=features_cfg["spectral"]["include_delta2"],
        )
        glottal, gnames = extract_glottal(
            audio, sr,
            voiced_only=features_cfg["glottal"]["voiced_only"],
            min_voiced_duration_seconds=features_cfg["glottal"]["min_voiced_duration_seconds"],
        )
        parts = [spectral, glottal]
        names = snames + gnames

    elif base_condition == "C3":
        glottal, gnames = extract_glottal(
            audio, sr,
            voiced_only=features_cfg["glottal"]["voiced_only"],
            min_voiced_duration_seconds=features_cfg["glottal"]["min_voiced_duration_seconds"],
        )
        parts = [glottal]
        names = gnames

    elif base_condition in ("C4", "C4_WavLM"):
        model_key = "wavlm" if base_condition == "C4_WavLM" else "hubert"
        emb = extract_ssl_embedding(
            audio, sr,
            model_key=model_key,
            cache_dir=cache_dir / model_key if cache_dir else None,
            audio_path=audio_path,
        )
        return emb, [f"{model_key}_emb_{i}" for i in range(len(emb))]

    else:
        raise ValueError(f"Unknown base condition: {base_condition}")

    return np.concatenate(parts).astype(np.float32), names


def _zero_vector(condition: str, features_cfg: dict) -> np.ndarray:
    """Return a zero vector of the expected dimensionality for a condition."""
    n_mfcc = features_cfg["spectral"]["n_mfcc"]
    spectral_dim = n_mfcc * 2 * (1
        + int(features_cfg["spectral"]["include_delta"])
        + int(features_cfg["spectral"]["include_delta2"])) + 6
    glottal_dim = 11
    prosodic_dim = 10  # approximate
    ssl_dim = 768

    dims = {
        "C1": prosodic_dim + spectral_dim + glottal_dim,
        "C2": spectral_dim + glottal_dim,
        "C3": glottal_dim,
        "C4": ssl_dim,
        "C4_WavLM": ssl_dim,
    }
    return np.zeros(dims.get(condition, spectral_dim + glottal_dim), dtype=np.float32)


def _get_expected_names(condition: str, features_cfg: dict) -> List[str]:
    """Return placeholder feature names for zero-vector fallback."""
    return [f"{condition}_feat_{i}" for i in range(len(_zero_vector(condition, features_cfg)))]


def build_feature_matrix(
    manifest_df: pd.DataFrame,
    condition: str,
    audio_cfg: dict,
    features_cfg: dict,
    models_cfg: dict,
    cache_dir: Optional[Path] = None,
    sentences: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build the full feature matrix X and label vector y for a manifest.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Recording manifest with ``filepath``, ``label``, and optionally
        ``language``, ``sentence_id`` columns.
    condition : str
        Feature condition identifier.
    audio_cfg : dict
        Audio config section.
    features_cfg : dict
        Features config section.
    models_cfg : dict
        Models config section.
    cache_dir : Path, optional
        Root cache directory.
    sentences : dict, optional
        ``sentences.json`` data; required for multimodal conditions.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Label vector.
    feature_names : list of str
        Feature names (from first successful extraction).
    """
    is_multimodal = condition.endswith("_MM")
    cache = FeatureCache(cache_dir / "audio" if cache_dir else Path(".cache/audio"),
                         features_cfg) if cache_dir else None

    X_rows = []
    y_rows = []
    feature_names: List[str] = []

    for _, row in manifest_df.iterrows():
        text_info = None
        if is_multimodal:
            if sentences is None:
                raise ValueError("sentences dict required for multimodal conditions.")
            lang = row.get("language", "EN")
            sid = row.get("sentence_id", "S1")
            text = sentences.get(lang, {}).get(sid, "")
            if not text or "[Yoruba" in text:
                logger.warning(
                    "Placeholder text for %s/%s; skipping multimodal for this recording.",
                    lang, sid
                )
                continue
            text_info = {"text": text, "language": lang, "sentence_id": sid}

        try:
            feats, fnames = build_feature_vector(
                audio_path=Path(row["filepath"]),
                condition=condition,
                audio_cfg=audio_cfg,
                features_cfg=features_cfg,
                models_cfg=models_cfg,
                cache=cache,
                cache_dir=cache_dir,
                text_info=text_info,
            )
            X_rows.append(feats)
            y_rows.append(int(row["label"]))
            if not feature_names:
                feature_names = fnames
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", row["filepath"], exc)

    if cache is not None:
        cache.report()

    if not X_rows:
        raise RuntimeError(f"No features extracted for condition {condition}.")

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    logger.info(
        "Built feature matrix: X=%s, y=%s for condition=%s.",
        X.shape, y.shape, condition
    )
    return X, y, feature_names
