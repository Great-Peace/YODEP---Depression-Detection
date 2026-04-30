"""Experiment: Cross-lingual transfer C1-XFER, C2-XFER, C3-XFER.

Train on English YODEP, test on Yoruba (and vice versa) using all classifiers.

Usage::

    python experiments/run_transfer.py

Outputs:
    - results/tables/table3_transfer.csv / .tex
    - results/figures/transfer_curves_C1_C2_C3.png
    - results/predictions/XFER_*.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.yodep_loader import load_yodep_manifest
from src.evaluation.metrics import compute_metrics
from src.features.pipeline import build_feature_matrix
from src.models.all_classifiers import build_clf_registry
from src.models.logistic_model import predict_logistic, train_logistic
from src.models.random_forest import predict_random_forest, train_random_forest
from src.models.svm_model import predict_svm, train_svm
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import set_all_seeds
from src.visualisation.results_tables import build_table3_transfer, save_table
from src.visualisation.transfer_curves import plot_transfer_curves


def main():
    project_root = Path(__file__).parent.parent
    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logging(
        log_dir=project_root / cfg["paths"]["logs"],
        log_filename="transfer.log"
    )
    logger = get_logger(__name__)
    set_all_seeds(cfg["project"]["random_seed"])

    results_dir  = project_root / cfg["paths"]["results"]
    figures_dir  = results_dir / "figures"
    pred_dir     = results_dir / "predictions"
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    yodep_raw     = project_root / cfg["paths"]["yodep_raw"]
    sentences_path = project_root / cfg["paths"]["sentences"]
    with open(sentences_path) as f:
        sentences = json.load(f)

    manifest_df = load_yodep_manifest(yodep_raw)
    if manifest_df.empty:
        logger.error("No YODEP data loaded.")
        sys.exit(1)

    cache_dir   = project_root / cfg["paths"]["cache"] / "features"
    audio_cfg   = cfg["audio"]
    features_cfg = cfg["features"]
    models_cfg  = cfg["models"]
    seed        = cfg["project"]["random_seed"]

    # All classifiers — same registry as main experiment
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

    conditions   = ["C1", "C2", "C3"]
    results_rows = []
    transfer_plot_rows = []

    for condition in conditions:
        logger.info("Transfer experiment: condition=%s", condition)

        en_df = manifest_df[manifest_df["language"] == "EN"].copy()
        yo_df = manifest_df[manifest_df["language"] == "YO"].copy()

        if en_df.empty or yo_df.empty:
            logger.warning("Skipping %s — missing EN or YO data.", condition)
            continue

        try:
            X_en, y_en, _ = build_feature_matrix(
                en_df, condition, audio_cfg, features_cfg, models_cfg,
                cache_dir=cache_dir, sentences=sentences
            )
            X_yo, y_yo, _ = build_feature_matrix(
                yo_df, condition, audio_cfg, features_cfg, models_cfg,
                cache_dir=cache_dir, sentences=sentences
            )
        except Exception as exc:
            logger.error("Feature extraction failed for %s: %s", condition, exc)
            continue

        if len(np.unique(y_en)) < 2 or len(np.unique(y_yo)) < 2:
            logger.warning("Insufficient class diversity for %s.", condition)
            continue

        cond_label = f"{condition}_XFER"

        for clf_name, (train_fn, predict_fn) in clf_registry.items():
            logger.info("  %s — classifier=%s", cond_label, clf_name)

            try:
                # EN → YO
                model_en      = train_fn(X_en, y_en)
                y_pred_en_yo, y_prob_en_yo = predict_fn(model_en, X_yo)
                m_en_yo       = compute_metrics(y_yo, y_pred_en_yo, y_prob_en_yo)

                # YO → EN
                model_yo      = train_fn(X_yo, y_yo)
                y_pred_yo_en, y_prob_yo_en = predict_fn(model_yo, X_en)
                m_yo_en       = compute_metrics(y_en, y_pred_yo_en, y_prob_yo_en)

            except Exception as exc:
                logger.warning("  Classifier %s failed for %s: %s", clf_name, condition, exc)
                continue

            f1_en_yo = m_en_yo["f1_macro"]
            f1_yo_en = m_yo_en["f1_macro"]

            # Save predictions
            for direction, y_true, y_pred, y_prob in [
                ("EN_to_YO", y_yo,  y_pred_en_yo, y_prob_en_yo),
                ("YO_to_EN", y_en,  y_pred_yo_en, y_prob_yo_en),
            ]:
                np.savez(
                    pred_dir / f"XFER_{condition}_{direction}_{clf_name}.npz",
                    y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                )

            # EN within-language: use the main LOSO F1 from table2 as baseline
            # Here we store cross-lingual results; delta is computed against
            # within-language LOSO values from table2_yodep_main.csv at report time
            for train_l, test_l, f1 in [
                ("EN", "YO", f1_en_yo),
                ("YO", "EN", f1_yo_en),
            ]:
                results_rows.append({
                    "Condition":    cond_label,
                    "Train_language": train_l,
                    "Test_language":  test_l,
                    "Classifier":   clf_name,
                    "F1_macro":     round(f1, 4),
                    "Delta_vs_within_language": float("nan"),  # filled below
                })
                transfer_plot_rows.append({
                    "condition":      cond_label,
                    "train_language": train_l,
                    "test_language":  test_l,
                    "classifier":     clf_name,
                    "f1_macro":       f1,
                })

            logger.info(
                "    EN→YO F1=%.4f | YO→EN F1=%.4f",
                f1_en_yo, f1_yo_en
            )

    # Fill delta against within-language LOSO baseline from table2
    table2_path = results_dir / "tables" / "table2_yodep_main.csv"
    if table2_path.exists():
        t2 = pd.read_csv(table2_path)
        within_map = {}
        for _, row in t2.iterrows():
            key = (row["Condition"], row["Language"], row["Classifier"])
            within_map[key] = row["F1_macro_mean"]

        for row in results_rows:
            cond_base = row["Condition"].replace("_XFER", "")
            train_l   = row["Train_language"]
            clf       = row["Classifier"]
            within_f1 = within_map.get((cond_base, train_l, clf), float("nan"))
            if not (np.isnan(row["F1_macro"]) or np.isnan(within_f1)):
                row["Delta_vs_within_language"] = round(row["F1_macro"] - within_f1, 4)

    if results_rows:
        table3 = build_table3_transfer(results_rows)
        save_table(
            table3,
            output_dir=results_dir / "tables",
            stem="table3_transfer",
            caption=(
                "Cross-lingual transfer results: all classifiers trained on one "
                "language and tested on the other. $\\Delta$ is relative to "
                "within-language LOSO baseline from Table 2."
            ),
            label="tab:transfer",
        )
        print(table3.to_string(index=False))

        transfer_df = pd.DataFrame(transfer_plot_rows)
        plot_transfer_curves(transfer_df, figures_dir)

    logger.info("Transfer experiments complete.")


if __name__ == "__main__":
    main()
