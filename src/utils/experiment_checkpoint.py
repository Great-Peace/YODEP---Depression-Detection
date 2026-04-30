"""Experiment-level checkpoint/resume for LOSO cross-validation runs.

Persists completed fold results to disk so that if a job is interrupted
(e.g. by a cloud provider restart), the run resumes from the last completed
fold rather than starting over.

Usage::

    ckpt = ExperimentCheckpoint(
        checkpoint_dir=Path(".cache/exp_ckpts"),
        run_id="yodep_C1_EN_svm",
    )
    for train_df, test_df, speaker_id in loso_folds(df):
        if ckpt.is_done(speaker_id):
            # already finished — reload cached result
            fold_result = ckpt.load_fold(speaker_id)
        else:
            fold_result = run_fold(...)
            ckpt.save_fold(speaker_id, fold_result)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ExperimentCheckpoint:
    """Persist and restore LOSO fold results across job restarts.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory where checkpoint JSON files are written.
    run_id : str
        Unique identifier for this experiment run (e.g. ``"C1_EN_svm"``).
        Each run gets its own subdirectory.

    Notes
    -----
    Each completed fold is written as a separate JSON file:
    ``{checkpoint_dir}/{run_id}/fold_{speaker_id}.json``.
    A manifest file ``manifest.json`` tracks which folds are done.
    """

    def __init__(self, checkpoint_dir: Path, run_id: str) -> None:
        self._dir = Path(checkpoint_dir) / run_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._dir / "manifest.json"
        self._manifest: Dict[str, bool] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, bool]:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                data = json.load(f)
            logger.info(
                "Loaded experiment checkpoint: %d folds done (%s).",
                sum(data.values()), self._dir.name,
            )
            return data
        return {}

    def _save_manifest(self) -> None:
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def is_done(self, speaker_id: str) -> bool:
        """Return True if this fold has already been completed.

        Parameters
        ----------
        speaker_id : str
            The held-out speaker identifier.

        Returns
        -------
        bool
        """
        return self._manifest.get(speaker_id, False)

    def save_fold(self, speaker_id: str, result: Dict[str, Any]) -> None:
        """Persist a completed fold result.

        Parameters
        ----------
        speaker_id : str
            Held-out speaker identifier.
        result : dict
            Fold metrics and predictions (must be JSON-serialisable or
            contain NumPy scalars/arrays which are handled automatically).
        """
        fold_path = self._dir / f"fold_{speaker_id}.json"

        # Strip numpy arrays from fold_predictions (too large for JSON)
        serialisable = {
            k: v for k, v in result.items()
            if k not in ("fold_predictions",)
        }

        with open(fold_path, "w") as f:
            json.dump(serialisable, f, indent=2, cls=_NumpyEncoder)

        self._manifest[speaker_id] = True
        self._save_manifest()
        logger.debug("Fold %s checkpointed → %s", speaker_id, fold_path)

    def load_fold(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Load a previously saved fold result.

        Parameters
        ----------
        speaker_id : str
            Held-out speaker identifier.

        Returns
        -------
        dict or None
            Loaded fold metrics, or None if not found.
        """
        fold_path = self._dir / f"fold_{speaker_id}.json"
        if not fold_path.exists():
            return None
        with open(fold_path) as f:
            data = json.load(f)
        logger.debug("Loaded checkpoint for fold %s.", speaker_id)
        return data

    def completed_folds(self) -> List[str]:
        """Return list of speaker IDs whose folds are done."""
        return [sid for sid, done in self._manifest.items() if done]

    def clear(self) -> None:
        """Delete all checkpoint files for this run."""
        import shutil
        if self._dir.exists():
            shutil.rmtree(self._dir)
        logger.info("Cleared experiment checkpoint: %s", self._dir)
