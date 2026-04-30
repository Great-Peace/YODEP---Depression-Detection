"""Verify YODEP folder structure before running experiments.

Checks file naming convention, counts, audio integrity, and minimum duration.

Usage::

    python scripts/verify_yodep.py
    python scripts/verify_yodep.py --raw-dir data/yodep/raw
    python scripts/verify_yodep.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_utils import load_audio
from src.data.yodep_loader import LABEL_MAP, parse_filename
from src.utils.logger import get_logger, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify YODEP wav file structure before running experiments. "
            "Checks naming convention, audio integrity, and minimum duration."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Path to YODEP raw directory (default: from config.yaml).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum duration in seconds (default: from config.yaml).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for each file.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logging(level=20)  # INFO
    logger = get_logger(__name__)

    raw_dir = args.raw_dir or (project_root / cfg["paths"]["yodep_raw"])
    min_duration = args.min_duration or cfg["audio"]["min_duration_seconds"]
    target_sr = cfg["audio"]["sample_rate"]

    if not raw_dir.exists():
        print(f"ERROR: Directory not found: {raw_dir}")
        print("Create the directory and place YODEP .wav files there.")
        sys.exit(1)

    wav_files = sorted(raw_dir.glob("*.wav"))
    print(f"\n{'='*60}")
    print(f"YODEP Verification — {raw_dir}")
    print(f"{'='*60}")
    print(f"Found {len(wav_files)} .wav files.\n")

    if len(wav_files) == 0:
        print("No .wav files found. Nothing to verify.")
        sys.exit(0)

    ok_count = 0
    errors = []
    warnings = []
    speakers = set()
    languages = set()
    conditions = set()
    durations = []

    for wav_path in wav_files:
        parsed = parse_filename(wav_path.name)
        if parsed is None:
            errors.append(
                f"  NAMING ERROR: {wav_path.name} — does not match "
                f"[SpeakerID]_[Language]_[Condition]_[Sentence]_[Take].wav"
            )
            continue

        speakers.add(parsed["speaker_id"])
        languages.add(parsed["language"])
        conditions.add(parsed["condition"])

        # Audio integrity check
        try:
            audio, sr = load_audio(wav_path, target_sr=target_sr)
            duration = len(audio) / sr
            durations.append(duration)

            if duration < min_duration:
                warnings.append(
                    f"  SHORT: {wav_path.name} — {duration:.2f}s "
                    f"(min={min_duration}s)"
                )
            else:
                ok_count += 1
                if args.verbose:
                    print(f"  OK: {wav_path.name} ({duration:.2f}s)")

        except Exception as exc:
            errors.append(f"  AUDIO ERROR: {wav_path.name} — {exc}")

    # Summary
    print(f"Speakers found: {sorted(speakers)}")
    print(f"Languages found: {sorted(languages)}")
    print(f"Conditions found: {sorted(conditions)}")
    if durations:
        import statistics
        print(f"\nDuration stats:")
        print(f"  Min:    {min(durations):.2f}s")
        print(f"  Max:    {max(durations):.2f}s")
        print(f"  Mean:   {statistics.mean(durations):.2f}s")
        print(f"  Median: {statistics.median(durations):.2f}s")

    # Expected structure check
    print(f"\nExpected structure check:")
    expected_per_speaker = len(cfg["yodep"]["languages"]) * len(cfg["yodep"]["conditions"]) * 5
    for spk in sorted(speakers):
        spk_files = [f for f in wav_files if f.name.startswith(spk + "_")]
        n = len(spk_files)
        status = "OK" if n == expected_per_speaker else f"WARNING: expected {expected_per_speaker}, got {n}"
        print(f"  {spk}: {n} files — {status}")

    print(f"\nResults:")
    print(f"  Valid files:   {ok_count}")
    print(f"  Warnings:      {len(warnings)}")
    print(f"  Errors:        {len(errors)}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(w)

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(e)
        print("\nFix the above errors before running experiments.")
        sys.exit(1)
    else:
        print("\nVerification passed — ready to run experiments.")
        sys.exit(0)


if __name__ == "__main__":
    main()
