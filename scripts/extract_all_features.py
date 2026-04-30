"""Extract all features for both datasets and cache to disk.

Usage::

    python scripts/extract_all_features.py
    python scripts/extract_all_features.py --dataset yodep
    python scripts/extract_all_features.py --dataset daic
    python scripts/extract_all_features.py --conditions C1 C2 C3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.pipeline import build_feature_matrix
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import set_all_seeds


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract and cache features for all recordings."
    )
    parser.add_argument(
        "--dataset",
        choices=["yodep", "daic", "both"],
        default="both",
        help="Which dataset to process.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["C1", "C2", "C3", "C4"],
        help="Feature conditions to extract.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logging(
        log_dir=project_root / cfg["paths"]["logs"],
        log_filename="extract_features.log"
    )
    logger = get_logger(__name__)
    set_all_seeds(cfg["project"]["random_seed"])

    cache_dir = project_root / cfg["paths"]["cache"] / "features"
    audio_cfg = cfg["audio"]
    features_cfg = cfg["features"]
    models_cfg = cfg["models"]

    sentences = None
    if (project_root / cfg["paths"]["sentences"]).exists():
        with open(project_root / cfg["paths"]["sentences"]) as f:
            sentences = json.load(f)

    if args.dataset in ("yodep", "both"):
        from src.data.yodep_loader import load_yodep_manifest

        yodep_raw = project_root / cfg["paths"]["yodep_raw"]
        if not yodep_raw.exists():
            logger.warning("YODEP raw dir not found: %s — skipping.", yodep_raw)
        else:
            manifest = load_yodep_manifest(yodep_raw)
            if not manifest.empty:
                for condition in args.conditions:
                    logger.info("Extracting YODEP features: condition=%s", condition)
                    try:
                        build_feature_matrix(
                            manifest, condition, audio_cfg, features_cfg, models_cfg,
                            cache_dir=cache_dir, sentences=sentences
                        )
                        logger.info("Done: YODEP condition=%s", condition)
                    except Exception as exc:
                        logger.error("Failed YODEP condition=%s: %s", condition, exc)

    if args.dataset in ("daic", "both"):
        from src.data.daic_loader import load_daic_woz

        daic_raw = project_root / cfg["paths"]["daic_woz_raw"]
        if not daic_raw.exists() or not any(daic_raw.iterdir()):
            logger.warning("DAIC-WOZ raw dir empty: %s — skipping.", daic_raw)
        else:
            try:
                train_df, dev_df = load_daic_woz(
                    daic_raw, phq8_threshold=cfg["daic_woz"]["phq8_threshold"]
                )
                import pandas as pd
                combined = pd.concat([train_df, dev_df], ignore_index=True)
                daic_conditions = [c for c in args.conditions if c in ["C1", "C3"]]
                for condition in daic_conditions:
                    logger.info("Extracting DAIC-WOZ features: condition=%s", condition)
                    try:
                        build_feature_matrix(
                            combined, condition, audio_cfg, features_cfg, models_cfg,
                            cache_dir=cache_dir
                        )
                        logger.info("Done: DAIC-WOZ condition=%s", condition)
                    except Exception as exc:
                        logger.error("Failed DAIC-WOZ condition=%s: %s", condition, exc)
            except Exception as exc:
                logger.error("DAIC-WOZ load failed: %s", exc)

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
