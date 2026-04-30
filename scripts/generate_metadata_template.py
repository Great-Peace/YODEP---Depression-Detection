"""Generate empty metadata.csv template with correct headers.

Usage::

    python scripts/generate_metadata_template.py
    python scripts/generate_metadata_template.py --output data/yodep/metadata.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create empty YODEP metadata.csv template with correct column headers."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: data/yodep/metadata.csv).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    out_path = args.output or (project_root / "data" / "yodep" / "metadata.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "speaker_id",          # e.g. P01
        "age_range",           # e.g. 18-25, 26-35
        "gender",              # M / F / Other
        "yoruba_native",       # yes / no
        "self_rated_acting_score",  # 1-5 scale (1=poor, 5=excellent)
        "has_voice_condition", # yes / no
        "session_date",        # YYYY-MM-DD
    ]

    df = pd.DataFrame(columns=columns)
    df.to_csv(out_path, index=False)

    print(f"Metadata template created: {out_path}")
    print(f"Columns: {', '.join(columns)}")
    print(
        "\nFill in one row per SPEAKER (not per recording). "
        "Speaker IDs must match the P01, P02, ... format used in filenames."
    )
    print("\nself_rated_acting_score: ask each participant 'How well do you think")
    print("  you performed the depressed condition?' on a 1–5 Likert scale.")


if __name__ == "__main__":
    main()
