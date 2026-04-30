"""Experiment: C1 and C3 on DAIC-WOZ — pipeline validation against published baselines.

Usage::

    python experiments/run_daic_validation.py

Outputs (saved to results/tables/):
    - table1_daic_validation.csv
    - table1_daic_validation.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.daic_loader import load_daic_woz
from src.evaluation.metrics import bootstrap_ci, compute_metrics
from src.features.pipeline import build_feature_matrix
from src.models.svm_model import predict_svm, train_svm
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import set_all_seeds
from src.visualisation.results_tables import build_table1_daic, save_table


def main():
    project_root = Path(__file__).parent.parent

    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    log_dir = project_root / cfg["paths"]["logs"]
    setup_logging(log_dir=log_dir, log_filename="daic_validation.log")
    logger = get_logger(__name__)

    set_all_seeds(cfg["project"]["random_seed"])

    # Save config snapshot
    results_dir = project_root / cfg["paths"]["results"]
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    with open(results_dir / "tables" / "daic_validation_config.yaml", "w") as f:
        yaml.dump(cfg, f)

    daic_raw = project_root / cfg["paths"]["daic_woz_raw"]

    if not daic_raw.exists() or not any(daic_raw.iterdir()):
        logger.error(
            "DAIC-WOZ raw data not found at %s. "
            "Please place the DAIC-WOZ files there.",
            daic_raw,
        )
        sys.exit(1)

    logger.info("Loading DAIC-WOZ data from %s", daic_raw)
    train_df, dev_df = load_daic_woz(
        raw_dir=daic_raw,
        phq8_threshold=cfg["daic_woz"]["phq8_threshold"],
    )

    cache_dir = project_root / cfg["paths"]["cache"] / "features"
    audio_cfg = cfg["audio"]
    features_cfg = cfg["features"]
    models_cfg = cfg["models"]
    svm_cfg = cfg["svm"]

    results_rows = []

    for condition in ["C1", "C3"]:
        logger.info("Running DAIC-WOZ validation: condition=%s", condition)

        try:
            X_train, y_train, fnames = build_feature_matrix(
                train_df, condition, audio_cfg, features_cfg, models_cfg,
                cache_dir=cache_dir
            )
            X_test, y_test, _ = build_feature_matrix(
                dev_df, condition, audio_cfg, features_cfg, models_cfg,
                cache_dir=cache_dir
            )
        except Exception as exc:
            logger.error("Feature extraction failed for %s: %s", condition, exc)
            continue

        model = train_svm(X_train, y_train, svm_cfg, cfg["project"]["random_seed"])
        y_pred, y_prob = predict_svm(model, X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        ci_low, ci_high = bootstrap_ci(
            y_test, y_pred, y_prob,
            n_iterations=cfg["evaluation"]["bootstrap_iterations"],
            random_seed=cfg["project"]["random_seed"],
        )

        results_rows.append({
            "Condition": condition,
            "Classifier": "SVM",
            "F1_macro": round(metrics["f1_macro"], 4),
            "Precision": round(metrics["precision_macro"], 4),
            "Recall": round(metrics["recall_macro"], 4),
            "Accuracy": round(metrics["accuracy"], 4),
            "AUC": round(metrics.get("auc", float("nan")), 4),
            "CI_lower_95": round(ci_low, 4),
            "CI_upper_95": round(ci_high, 4),
        })

        logger.info(
            "DAIC-WOZ %s SVM — F1=%.4f, AUC=%.4f, CI=[%.4f, %.4f]",
            condition, metrics["f1_macro"], metrics.get("auc", 0),
            ci_low, ci_high
        )

    if results_rows:
        table1 = build_table1_daic(results_rows)
        save_table(
            table1,
            output_dir=results_dir / "tables",
            stem="table1_daic_validation",
            caption=(
                "DAIC-WOZ Validation: F1 macro, AUC, and 95\\% bootstrap CI "
                "for SVM classifier under C1 (full features) and "
                "C3 (glottal only) conditions."
            ),
            label="tab:daic_validation",
        )
        print(table1.to_string(index=False))
    else:
        logger.error("No results produced.")

    logger.info("DAIC-WOZ validation complete.")


if __name__ == "__main__":
    main()
