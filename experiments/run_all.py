"""Master script: runs every experiment in sequence and saves all results.

Usage::

    python experiments/run_all.py                   # YODEP main + transfer + figures
    python experiments/run_all.py --include-daic    # also run DAIC-WOZ validation
    python experiments/run_all.py --skip-transfer   # skip cross-lingual transfer
    python experiments/run_all.py --skip-figures    # skip figure generation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, setup_logging


def run_script(script_path: Path, args: list = None):
    cmd = [sys.executable, str(script_path)] + (args or [])
    logger = get_logger(__name__)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Script failed with exit code %d: %s", result.returncode, script_path)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all YODEP experiments.")
    parser.add_argument("--include-daic", action="store_true",
                        help="Also run DAIC-WOZ validation (skipped by default).")
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip cross-lingual transfer experiment.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation after experiments complete.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logging(
        log_dir=project_root / cfg["paths"]["logs"],
        log_filename="run_all.log"
    )
    logger = get_logger(__name__)
    logger.info("=== YODEP: Running all experiments ===")

    exp_dir = Path(__file__).parent
    scripts_dir = project_root / "scripts"
    exit_codes = []

    if args.include_daic:
        ec = run_script(exp_dir / "run_daic_validation.py")
        exit_codes.append(("daic_validation", ec))

    ec = run_script(exp_dir / "run_yodep_main.py")
    exit_codes.append(("yodep_main", ec))

    if not args.skip_transfer:
        ec = run_script(exp_dir / "run_transfer.py")
        exit_codes.append(("transfer", ec))

    # F0 analysis scripts (extras 1, 2, 3)
    f0_script = scripts_dir / "run_f0_analysis.py"
    if f0_script.exists():
        ec = run_script(f0_script)
        exit_codes.append(("f0_analysis", ec))

    # Generate all publication-quality figures from results
    if not args.skip_figures:
        fig_script = scripts_dir / "generate_report_figures.py"
        if fig_script.exists():
            logger.info("Generating report figures...")
            ec = run_script(fig_script)
            exit_codes.append(("generate_figures", ec))
        else:
            logger.warning("Figure script not found: %s", fig_script)

    logger.info("=== All experiments complete ===")
    for name, ec in exit_codes:
        status = "OK" if ec == 0 else f"FAILED (exit={ec})"
        logger.info("  %-25s %s", name, status)

    n_failed = sum(1 for _, ec in exit_codes if ec != 0)
    if n_failed > 0:
        logger.warning("%d experiment(s) failed.", n_failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
