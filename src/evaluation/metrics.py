"""Evaluation metrics: F1, precision, recall, AUC, bootstrap CI.

All metrics are computed per fold and aggregated as mean ± SD.
Bootstrap confidence intervals use 1000 iterations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray, optional
        Predicted probabilities, shape (n_samples, n_classes).  Required
        for AUC.

    Returns
    -------
    dict
        Keys: ``f1_macro``, ``f1_weighted``, ``precision_macro``,
        ``recall_macro``, ``accuracy``, ``auc`` (if y_prob provided).
    """
    metrics: Dict[str, float] = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            metrics["auc"] = auc
        except Exception as exc:
            logger.warning("AUC computation failed: %s", exc)
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")

    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    metric: str = "f1_macro",
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray, optional
        Predicted probabilities.
    metric : str
        Metric key from :func:`compute_metrics`.
    n_iterations : int
        Number of bootstrap samples.
    confidence_level : float
        Confidence level (e.g. 0.95 for 95% CI).
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    ci_lower : float
    ci_upper : float

    Notes
    -----
    Uses the percentile bootstrap method.  Samples with replacement from
    (y_true, y_pred) pairs.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    scores = []

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        yprob = y_prob[idx] if y_prob is not None else None

        if len(np.unique(yt)) < 2:
            continue

        m = compute_metrics(yt, yp, yprob)
        scores.append(m.get(metric, float("nan")))

    scores = [s for s in scores if not np.isnan(s)]
    if not scores:
        return float("nan"), float("nan")

    alpha = (1 - confidence_level) / 2
    ci_lower = float(np.percentile(scores, alpha * 100))
    ci_upper = float(np.percentile(scores, (1 - alpha) * 100))
    return ci_lower, ci_upper


def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-fold metrics into mean and standard deviation.

    Parameters
    ----------
    fold_metrics : list of dict
        Each element is the output of :func:`compute_metrics` for one fold.

    Returns
    -------
    dict
        Keys follow the pattern ``{metric}_mean`` and ``{metric}_std``.
    """
    keys = fold_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if isinstance(m.get(k), (int, float)) and not np.isnan(float(m[k]))]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    return agg
