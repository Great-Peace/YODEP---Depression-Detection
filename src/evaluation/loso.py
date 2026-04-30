"""Leave-One-Speaker-Out cross-validation runner and aggregator.

Supports checkpoint/resume: completed folds are persisted to disk so that
interrupted jobs (e.g. cloud provider restarts) resume from the last
completed fold rather than restarting from scratch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.yodep_loader import loso_folds
from ..features.pipeline import build_feature_matrix
from ..utils.experiment_checkpoint import ExperimentCheckpoint
from ..utils.logger import get_logger
from .metrics import aggregate_fold_metrics, bootstrap_ci, compute_metrics

logger = get_logger(__name__)


def _make_run_id(condition: str, language: Optional[str], classifier_name: str) -> str:
    lang = language or "ALL"
    return f"{condition}__{lang}__{classifier_name}"


def run_loso(
    manifest_df: pd.DataFrame,
    condition: str,
    train_fn: Callable,
    predict_fn: Callable,
    audio_cfg: dict,
    features_cfg: dict,
    models_cfg: dict,
    language: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    sentences: Optional[Dict] = None,
    bootstrap_iterations: int = 1000,
    random_seed: int = 42,
    classifier_name: str = "clf",
    checkpoint_dir: Optional[Path] = None,
) -> Dict:
    """Run Leave-One-Speaker-Out CV for a given condition and classifier.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Full YODEP manifest.
    condition : str
        Feature condition (e.g. ``"C1"``, ``"C2_MM"``).
    train_fn : callable
        Function ``(X_train, y_train) -> model``.
    predict_fn : callable
        Function ``(model, X_test) -> (y_pred, y_prob)``.
    audio_cfg : dict
        Audio config section.
    features_cfg : dict
        Features config section.
    models_cfg : dict
        Models config section.
    language : str, optional
        Filter to ``"EN"`` or ``"YO"``.  If *None*, uses all languages.
    cache_dir : Path, optional
        Feature cache directory.
    sentences : dict, optional
        Sentence stimuli dict; required for multimodal conditions.
    bootstrap_iterations : int
        Bootstrap CI iterations.
    random_seed : int
        Random seed.
    classifier_name : str
        Short name for the classifier used in checkpoint IDs (e.g. ``"svm"``).
    checkpoint_dir : Path, optional
        Directory for fold-level checkpoints.  If provided, completed folds
        are saved and reloaded on resume.  Pass ``None`` to disable.

    Returns
    -------
    dict
        Keys:
        - ``"fold_metrics"``: list of per-fold metric dicts
        - ``"fold_predictions"``: list of (y_true, y_pred, y_prob, speaker_id)
        - ``"aggregated"``: mean ± SD of all metrics
        - ``"ci_lower"`` / ``"ci_upper"``: bootstrap CI on F1_macro
        - ``"condition"``, ``"language"``
        - ``"feature_names"``: list of feature names from last successful fold

    Notes
    -----
    Checkpoint resume: if *checkpoint_dir* is given, each completed fold is
    written to ``{checkpoint_dir}/{condition}__{language}__{classifier_name}/
    fold_{speaker_id}.json``.  On restart, completed folds are skipped and
    their results reloaded.
    """
    fold_metrics: List[Dict] = []
    fold_predictions = []
    fnames: List[str] = []

    # Initialise checkpoint manager
    ckpt: Optional[ExperimentCheckpoint] = None
    if checkpoint_dir is not None:
        run_id = _make_run_id(condition, language, classifier_name)
        ckpt = ExperimentCheckpoint(checkpoint_dir, run_id)
        done = ckpt.completed_folds()
        if done:
            logger.info(
                "Resuming LOSO run %s — %d fold(s) already done: %s",
                run_id, len(done), done,
            )

    for train_df, test_df, speaker_id in loso_folds(manifest_df, language=language):
        # Resume: reload cached fold result
        if ckpt is not None and ckpt.is_done(speaker_id):
            cached = ckpt.load_fold(speaker_id)
            if cached is not None:
                logger.info("LOSO fold %s — loaded from checkpoint.", speaker_id)
                fold_metrics.append(cached.get("metrics", {}))
                # fold_predictions not stored in checkpoint (large arrays)
                continue

        logger.info("LOSO fold: held-out speaker=%s", speaker_id)

        try:
            X_train, y_train, fnames = build_feature_matrix(
                train_df, condition, audio_cfg, features_cfg, models_cfg,
                cache_dir=cache_dir, sentences=sentences
            )
            X_test, y_test, _ = build_feature_matrix(
                test_df, condition, audio_cfg, features_cfg, models_cfg,
                cache_dir=cache_dir, sentences=sentences
            )
        except Exception as exc:
            logger.warning("Feature extraction failed for fold %s: %s", speaker_id, exc)
            continue

        if len(np.unique(y_train)) < 2:
            logger.warning("Fold %s: only one class in training set; skipping.", speaker_id)
            continue

        model = train_fn(X_train, y_train)
        y_pred, y_prob = predict_fn(model, X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["speaker_id"] = speaker_id
        fold_metrics.append(metrics)
        fold_predictions.append((y_test, y_pred, y_prob, speaker_id))

        logger.info(
            "  Fold %s — F1_macro=%.4f, AUC=%.4f",
            speaker_id, metrics["f1_macro"], metrics.get("auc", float("nan"))
        )

        # Persist fold result
        if ckpt is not None:
            ckpt.save_fold(speaker_id, {"metrics": metrics})

    if not fold_metrics:
        logger.error("No folds completed for condition=%s, language=%s.", condition, language)
        return {
            "fold_metrics": [],
            "fold_predictions": [],
            "aggregated": {},
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "condition": condition,
            "language": language,
            "feature_names": [],
        }

    agg = aggregate_fold_metrics(fold_metrics)

    # Bootstrap CI — only possible when fold_predictions are in memory
    if fold_predictions:
        all_y_true = np.concatenate([fp[0] for fp in fold_predictions])
        all_y_pred = np.concatenate([fp[1] for fp in fold_predictions])
        all_y_prob = np.concatenate([fp[2] for fp in fold_predictions])
        ci_lower, ci_upper = bootstrap_ci(
            all_y_true, all_y_pred, all_y_prob,
            n_iterations=bootstrap_iterations, random_seed=random_seed
        )
    else:
        ci_lower, ci_upper = float("nan"), float("nan")

    logger.info(
        "LOSO complete — condition=%s, language=%s — "
        "F1_macro=%.4f±%.4f, CI=[%.4f, %.4f]",
        condition, language,
        agg.get("f1_macro_mean", float("nan")),
        agg.get("f1_macro_std", float("nan")),
        ci_lower, ci_upper,
    )

    return {
        "fold_metrics": fold_metrics,
        "fold_predictions": fold_predictions,
        "aggregated": agg,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "condition": condition,
        "language": language,
        "feature_names": fnames,
    }
