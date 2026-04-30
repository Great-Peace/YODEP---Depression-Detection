"""Normalize YODEP filenames from field recording convention to pipeline convention.

Converts from:
    YODEP/P001/P001_ENG_NORM_S1.m4a    (subfolders, 3-digit speaker ID, m4a)
    YODEP/P001/P001_ENG_NORM_S1.wav    (subfolders, 3-digit speaker ID, wav)
To:
    data/yodep/raw/P01_EN_NORMAL_S1_T1.wav   (flat directory, 2-digit ID, wav)

Mapping:
    Speaker : P001 → P01  (strips leading zero, keeps 2-digit zero-padded)
    Language: ENG  → EN   |  YOR → YO
    Condition: NORM → NORMAL  |  DEP → DEPRESSED
    Take    : appends _T1 (single take per sentence)

m4a files are converted to 16kHz mono wav using ffmpeg (must be installed).

Usage::

    # Dry run — prints what would happen, copies nothing
    python scripts/normalize_yodep_filenames.py --src ~/yodep_raw_download --dry-run

    # Live run — copies/converts renamed files to data/yodep/raw/
    python scripts/normalize_yodep_filenames.py --src ~/yodep_raw_download

    # Custom destination
    python scripts/normalize_yodep_filenames.py --src ~/yodep_raw_download --dst /data/yodep/raw
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac"}


# ── Normalisation maps ────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "ENG": "EN",
    "YOR": "YO",
    "EN":  "EN",   # already correct
    "YO":  "YO",
}

CONDITION_MAP = {
    "NORM":      "NORMAL",
    "NORMAL":    "NORMAL",
    "DEP":       "DEPRESSED",
    "DEPRESSED": "DEPRESSED",
}


def normalise_speaker(raw: str) -> str:
    """Convert P001 → P01 (strip leading zero, keep 2-digit zero-padded).

    Parameters
    ----------
    raw : str
        Raw speaker token e.g. ``"P001"``, ``"P01"``, ``"P10"``.

    Returns
    -------
    str
        Normalised speaker ID e.g. ``"P01"``, ``"P10"``.
    """
    digits = raw.lstrip("Pp").lstrip("0") or "0"
    return f"P{int(digits):02d}"


def parse_raw_filename(stem: str) -> dict | None:
    """Parse a raw YODEP filename stem into components.

    Parameters
    ----------
    stem : str
        Filename without extension, e.g. ``"P001_ENG_NORM_S1"`` or
        ``"P001_YOR_DEP_S3"``.

    Returns
    -------
    dict or None
        Keys: ``speaker_id``, ``language``, ``condition``, ``sentence_id``.
        Returns *None* if the stem cannot be parsed.
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    speaker_raw  = parts[0]               # P001
    language_raw = parts[1].upper()       # ENG / YOR
    condition_raw = parts[2].upper()      # NORM / DEP
    sentence_id   = parts[3].upper()      # S1 … S5

    if language_raw not in LANGUAGE_MAP:
        return None
    if condition_raw not in CONDITION_MAP:
        return None
    if not sentence_id.startswith("S"):
        return None

    return {
        "speaker_id":  normalise_speaker(speaker_raw),
        "language":    LANGUAGE_MAP[language_raw],
        "condition":   CONDITION_MAP[condition_raw],
        "sentence_id": sentence_id,
    }


def build_target_name(parsed: dict, take: str = "T1") -> str:
    """Build the pipeline-convention filename.

    Parameters
    ----------
    parsed : dict
        Output of :func:`parse_raw_filename`.
    take : str
        Take identifier.  Defaults to ``"T1"``.

    Returns
    -------
    str
        e.g. ``"P01_EN_NORMAL_S1_T1.wav"``
    """
    return (
        f"{parsed['speaker_id']}_"
        f"{parsed['language']}_"
        f"{parsed['condition']}_"
        f"{parsed['sentence_id']}_"
        f"{take}.wav"
    )


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _convert_to_wav(src: Path, dst: Path) -> None:
    """Convert any audio file to 16kHz mono WAV using ffmpeg.

    Parameters
    ----------
    src : Path
        Source audio file (e.g. .m4a).
    dst : Path
        Destination .wav file path.

    Raises
    ------
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",           # overwrite without asking
            "-i", str(src),           # input
            "-ar", "16000",           # resample to 16kHz
            "-ac", "1",               # mono
            "-sample_fmt", "s16",     # 16-bit PCM
            str(dst),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))


def run(src_dir: Path, dst_dir: Path, dry_run: bool = False) -> None:
    """Scan *src_dir* recursively, rename and copy/convert all audio files to *dst_dir*.

    Parameters
    ----------
    src_dir : Path
        Root of the downloaded YODEP folder (contains P001/, P002/, …).
    dst_dir : Path
        Target directory (``data/yodep/raw/``).  Created if needed.
    dry_run : bool
        If *True*, only prints what would happen without copying anything.
    """
    src_dir = Path(src_dir).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()

    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}")
        sys.exit(1)

    # Gather all supported audio files
    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(src_dir.rglob(f"*{ext}"))
        audio_files.extend(src_dir.rglob(f"*{ext.upper()}"))
    audio_files = sorted(set(audio_files))

    if not audio_files:
        print(f"ERROR: No audio files found under {src_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    # Check ffmpeg availability for non-wav files
    needs_ffmpeg = any(f.suffix.lower() != ".wav" for f in audio_files)
    has_ffmpeg = _check_ffmpeg()
    if needs_ffmpeg and not has_ffmpeg and not dry_run:
        print("ERROR: ffmpeg is required to convert m4a/mp3 files but was not found.")
        print("Install with: sudo apt-get install -y ffmpeg")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) under {src_dir}")
    ext_counts = {}
    for f in audio_files:
        ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count} file(s)")

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = errors = 0

    for src_path in audio_files:
        parsed = parse_raw_filename(src_path.stem)
        if parsed is None:
            print(f"  SKIP (unrecognised name): {src_path.name}")
            skipped += 1
            continue

        target_name = build_target_name(parsed)
        dst_path = dst_dir / target_name
        needs_conversion = src_path.suffix.lower() != ".wav"
        action_label = "CONVERT+RENAME" if needs_conversion else "RENAME+COPY"
        action = action_label if not dry_run else f"WOULD {action_label}"

        print(f"  {action}: {src_path.relative_to(src_dir)}  →  {target_name}")

        if not dry_run:
            if dst_path.exists():
                print(f"    WARNING: destination exists, overwriting.")
            try:
                if needs_conversion:
                    _convert_to_wav(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                ok += 1
            except Exception as exc:
                print(f"    ERROR: {exc}")
                errors += 1
        else:
            ok += 1

    print()
    print("─" * 60)
    print(f"{'DRY RUN — ' if dry_run else ''}Results:")
    print(f"  Processed : {ok}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    if not dry_run and ok > 0:
        print(f"\nFiles written to: {dst_dir}")
        print("\nNext step: python scripts/verify_yodep.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise YODEP filenames from field recording convention "
                    "to pipeline convention and copy to data/yodep/raw/."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source directory containing downloaded YODEP files "
             "(e.g. ~/yodep_raw_download).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination directory. Defaults to data/yodep/raw/ relative "
             "to the project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without copying any files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    dst = args.dst or (project_root / "data" / "yodep" / "raw")

    run(src_dir=args.src, dst_dir=dst, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
