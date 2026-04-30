"""Experiment: All conditions C1-C4 and multimodal on YODEP with LOSO CV.

Usage::

    python experiments/run_yodep_main.py [--language EN|YO|ALL]

Outputs (saved to results/tables/):
    - table2_yodep_main.csv / .tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.yodep_loader import load_yodep_manifest
from src.evaluation.loso import run_loso
from src.features.pipeline import build_feature_matrix
from src.models.all_classifiers import build_clf_registry
from src.models.logistic_model import predict_logistic, train_logistic
from src.models.mlp_fusion import MLPClassifier
from src.models.random_forest import (
    get_feature_importances,
    predict_random_forest,
    train_random_forest,
)
from src.models.svm_model import predict_svm, train_svm
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import set_all_seeds
from src.visualisation.feature_importance import plot_feature_importance
from src.visualisation.results_tables import build_table2_yodep_main, save_table


def main():
    parser = argparse.ArgumentParser(description="Run YODEP main experiments.")
    parser.add_argument(
        "--language", choices=["EN", "YO", "ALL"], default="ALL",
        help="Language subset to evaluate."
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=["C1", "C2", "C3", "C4"],
        help="Conditions to run."
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logging(
        log_dir=project_root / cfg["paths"]["logs"],
        log_filename="yodep_main.log"
    )
    logger = get_logger(__name__)
    set_all_seeds(cfg["project"]["random_seed"])

    results_dir = project_root / cfg["paths"]["results"]
    figures_dir = results_dir / "figures"
    predictions_dir = results_dir / "predictions"
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    yodep_raw = project_root / cfg["paths"]["yodep_raw"]
    yodep_meta = project_root / cfg["paths"]["yodep_raw"].replace("raw", "") / "metadata.csv"
    sentences_path = project_root / cfg["paths"]["sentences"]

    if not yodep_raw.exists():
        logger.error("YODEP raw data not found at %s.", yodep_raw)
        sys.exit(1)

    with open(sentences_path) as f:
        sentences = json.load(f)

    metadata_csv = Path(yodep_meta) if Path(yodep_meta).exists() else None
    manifest_df = load_yodep_manifest(yodep_raw, metadata_csv=metadata_csv)

    if manifest_df.empty:
        logger.error("No YODEP data loaded. Place .wav files in %s.", yodep_raw)
        sys.exit(1)

    cache_dir = project_root / cfg["paths"]["cache"] / "features"
    audio_cfg = cfg["audio"]
    features_cfg = cfg["features"]
    models_cfg = cfg["models"]
    seed = cfg["project"]["random_seed"]

    languages = (
        cfg["yodep"]["languages"]
        if args.language == "ALL"
        else [args.language]
    )
    languages = languages + ["ALL"]  # also run on combined

    handcrafted_conditions = [c for c in args.conditions if c in ["C1", "C2", "C3"]]
    ssl_conditions = [c for c in args.conditions if c in ["C4", "C4_WavLM"]]

    # All classifiers: original three (kept for feature-importance compat) + extended set
    clf_registry = {
        "svm": (
            lambda Xtr, ytr: train_svm(Xtr, ytr, cfg["svm"], seed),
            predict_svm,
        ),
        "logistic": (
            lambda Xtr, ytr: train_logistic(Xtr, ytr, cfg["logistic"], seed),
            predict_logistic,
        ),
        "random_forest": (
            lambda Xtr, ytr: train_random_forest(Xtr, ytr, cfg["random_forest"], seed),
            predict_random_forest,
        ),
        **build_clf_registry(
            cv_folds=cfg.get("classifiers", {}).get("cv_folds", 5),
            random_seed=seed,
        ),
    }

    results_rows = []
    fold_metrics_rows: List[Dict] = []
    rf_models_for_importance: Dict = {}
    training_curves_dir = results_dir / "training_curves"
    training_curves_dir.mkdir(parents=True, exist_ok=True)

    for condition in handcrafted_conditions:
        for lang in languages:
            lang_filter = None if lang == "ALL" else lang
            for clf_name, (train_fn, predict_fn) in clf_registry.items():
                logger.info(
                    "Running YODEP LOSO: condition=%s, language=%s, classifier=%s",
                    condition, lang, clf_name
                )
                result = run_loso(
                    manifest_df=manifest_df,
                    condition=condition,
                    train_fn=train_fn,
                    predict_fn=predict_fn,
                    audio_cfg=audio_cfg,
                    features_cfg=features_cfg,
                    models_cfg=models_cfg,
                    language=lang_filter,
                    cache_dir=cache_dir,
                    sentences=sentences,
                    bootstrap_iterations=cfg["evaluation"]["bootstrap_iterations"],
                    random_seed=seed,
                )
                agg = result.get("aggregated", {})
                results_rows.append({
                    "Condition": condition,
                    "Modality": "audio",
                    "Language": lang,
                    "Classifier": clf_name,
                    "F1_macro_mean": round(agg.get("f1_macro_mean", float("nan")), 4),
                    "F1_macro_std": round(agg.get("f1_macro_std", float("nan")), 4),
                    "Precision_mean": round(agg.get("precision_macro_mean", float("nan")), 4),
                    "Recall_mean": round(agg.get("recall_macro_mean", float("nan")), 4),
                    "Accuracy_mean": round(agg.get("accuracy_mean", float("nan")), 4),
                    "AUC_mean": round(agg.get("auc_mean", float("nan")), 4),
                })

                # Accumulate per-fold metrics for training/validation plots
                for fm in result.get("fold_metrics", []):
                    fold_metrics_rows.append({
                        "Condition": condition,
                        "Language": lang,
                        "Classifier": clf_name,
                        "Speaker": fm.get("speaker_id", ""),
                        "F1_macro": round(fm.get("f1_macro", float("nan")), 4),
                        "AUC": round(fm.get("auc", float("nan")), 4),
                        "Accuracy": round(fm.get("accuracy", float("nan")), 4),
                        "Precision": round(fm.get("precision_macro", float("nan")), 4),
                        "Recall": round(fm.get("recall_macro", float("nan")), 4),
                    })

                # Save fold-level predictions for ROC curves / confusion matrices
                fold_preds = result.get("fold_predictions", [])
                if fold_preds:
                    try:
                        pred_path = predictions_dir / f"{condition}_{lang}_{clf_name}.npz"
                        np.savez(
                            pred_path,
                            y_true=np.concatenate([fp[0] for fp in fold_preds]),
                            y_pred=np.concatenate([fp[1] for fp in fold_preds]),
                            y_prob=np.concatenate([fp[2] for fp in fold_preds]),
                        )
                    except Exception as exc:
                        logger.warning("Could not save predictions for %s/%s/%s: %s", condition, lang, clf_name, exc)

                # Save RF models for feature importance (Extra 4)
                if clf_name == "random_forest" and lang != "ALL":
                    rf_models_for_importance[(condition, lang)] = result

    # SSL conditions — all classifiers (same registry as handcrafted)
    for condition in ssl_conditions:
        for lang in languages:
            lang_filter = None if lang == "ALL" else lang
            for clf_name, (train_fn, predict_fn) in clf_registry.items():
                logger.info(
                    "Running YODEP LOSO: condition=%s, language=%s, classifier=%s",
                    condition, lang, clf_name
                )
                result = run_loso(
                    manifest_df=manifest_df,
                    condition=condition,
                    train_fn=train_fn,
                    predict_fn=predict_fn,
                    audio_cfg=audio_cfg,
                    features_cfg=features_cfg,
                    models_cfg=models_cfg,
                    language=lang_filter,
                    cache_dir=cache_dir,
                    sentences=sentences,
                    bootstrap_iterations=cfg["evaluation"]["bootstrap_iterations"],
                    random_seed=seed,
                )
                agg = result.get("aggregated", {})
                results_rows.append({
                    "Condition": condition,
                    "Modality": "ssl",
                    "Language": lang,
                    "Classifier": clf_name,
                    "F1_macro_mean": round(agg.get("f1_macro_mean", float("nan")), 4),
                    "F1_macro_std": round(agg.get("f1_macro_std", float("nan")), 4),
                    "Precision_mean": round(agg.get("precision_macro_mean", float("nan")), 4),
                    "Recall_mean": round(agg.get("recall_macro_mean", float("nan")), 4),
                    "Accuracy_mean": round(agg.get("accuracy_mean", float("nan")), 4),
                    "AUC_mean": round(agg.get("auc_mean", float("nan")), 4),
                })

                # Accumulate per-fold metrics
                for fm in result.get("fold_metrics", []):
                    fold_metrics_rows.append({
                        "Condition": condition,
                        "Language": lang,
                        "Classifier": clf_name,
                        "Speaker": fm.get("speaker_id", ""),
                        "F1_macro": round(fm.get("f1_macro", float("nan")), 4),
                        "AUC": round(fm.get("auc", float("nan")), 4),
                        "Accuracy": round(fm.get("accuracy", float("nan")), 4),
                        "Precision": round(fm.get("precision_macro", float("nan")), 4),
                        "Recall": round(fm.get("recall_macro", float("nan")), 4),
                    })

                # Save fold-level predictions
                fold_preds = result.get("fold_predictions", [])
                if fold_preds:
                    try:
                        pred_path = predictions_dir / f"{condition}_{lang}_{clf_name}.npz"
                        np.savez(
                            pred_path,
                            y_true=np.concatenate([fp[0] for fp in fold_preds]),
                            y_pred=np.concatenate([fp[1] for fp in fold_preds]),
                            y_prob=np.concatenate([fp[2] for fp in fold_preds]),
                        )
                    except Exception as exc:
                        logger.warning("Could not save predictions for %s/%s/%s: %s", condition, lang, clf_name, exc)

    # Feature importance plots (Extra 4)
    for (cond, lang), loso_result in rf_models_for_importance.items():
        fnames = loso_result.get("feature_names", [])
        if not fnames:
            continue
        # Use last fold's model (representative)
        fold_preds = loso_result.get("fold_predictions", [])
        if fold_preds:
            # Re-train once on all data for importance extraction
            try:
                from src.features.pipeline import build_feature_matrix
                lang_filter = None if lang == "ALL" else lang
                subset = manifest_df if lang_filter is None else manifest_df[manifest_df["language"] == lang_filter]
                X_all, y_all, fnames_all = build_feature_matrix(
                    subset, cond, audio_cfg, features_cfg, models_cfg,
                    cache_dir=cache_dir, sentences=sentences
                )
                rf_model = train_random_forest(X_all, y_all, cfg["random_forest"], seed)
                imp_df = get_feature_importances(rf_model, fnames_all)
                plot_feature_importance(imp_df, cond, lang, figures_dir)
            except Exception as exc:
                logger.warning("Feature importance plot failed for %s/%s: %s", cond, lang, exc)

    # Save per-fold metrics CSV for training/validation plots
    if fold_metrics_rows:
        import pandas as pd
        fold_df = pd.DataFrame(fold_metrics_rows)
        fold_df.to_csv(results_dir / "tables" / "fold_metrics_detail.csv", index=False)
        logger.info("Per-fold metrics saved (%d rows).", len(fold_df))

    # Save training curves for MLP and GBM classifiers (full-data retrain)
    curve_clfs = {k: v for k, v in clf_registry.items()
                  if k in ("mlp_sklearn", "gradient_boosting", "hist_gradient_boosting")}
    if curve_clfs:
        logger.info("Saving training curves for MLP / GBM classifiers...")
        for cond in handcrafted_conditions + ssl_conditions:
            for lang in [l for l in languages if l != "ALL"]:
                lang_filter = lang
                subset = manifest_df[manifest_df["language"] == lang_filter]
                try:
                    X_all, y_all, _ = build_feature_matrix(
                        subset, cond, audio_cfg, features_cfg, models_cfg,
                        cache_dir=cache_dir, sentences=sentences
                    )
                except Exception:
                    continue
                for clf_name, (train_fn, _) in curve_clfs.items():
                    try:
                        model     = train_fn(X_all, y_all)
                        best_clf  = model.best_estimator_.named_steps.get("clf")
                        if best_clf is None:
                            continue
                        curve_data: Dict = {}
                        if hasattr(best_clf, "loss_curve_"):
                            curve_data["loss"] = [float(v) for v in best_clf.loss_curve_]
                        if hasattr(best_clf, "validation_scores_"):
                            curve_data["val_score"] = [float(v) for v in best_clf.validation_scores_]
                        if hasattr(best_clf, "train_score_"):
                            curve_data["train_score"] = [float(v) for v in best_clf.train_score_]
                        if curve_data:
                            out = training_curves_dir / f"{cond}_{lang}_{clf_name}.json"
                            with open(out, "w") as fp:
                                json.dump(curve_data, fp)
                            logger.info("  Training curve saved: %s", out.name)
                    except Exception as exc:
                        logger.warning("Training curve failed %s/%s/%s: %s", cond, lang, clf_name, exc)

    if results_rows:
        import pandas as pd
        table2 = build_table2_yodep_main(results_rows)
        save_table(
            table2,
            output_dir=results_dir / "tables",
            stem="table2_yodep_main",
            caption=(
                "YODEP Main Results: F1 macro (mean $\\pm$ SD over LOSO folds) "
                "for all conditions and classifiers."
            ),
            label="tab:yodep_main",
        )
        print(table2.to_string(index=False))

    logger.info("YODEP main experiment complete.")


if __name__ == "__main__":
    main()
