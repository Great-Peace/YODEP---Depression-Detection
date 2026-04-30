"""Logistic Regression classifier with regularisation search."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_logistic(
    C_values: List[float] = None,
    max_iter: int = 1000,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> GridSearchCV:
    """Build a grid-search Logistic Regression pipeline.

    Parameters
    ----------
    C_values : list of float, optional
        Inverse regularisation strengths.  Defaults to
        ``[0.01, 0.1, 1, 10, 100]``.
    max_iter : int
        Maximum iterations for the solver.
    cv_folds : int
        Inner CV folds for grid search.
    random_seed : int
        Random seed.

    Returns
    -------
    GridSearchCV
        Unfitted grid search estimator.

    Notes
    -----
    Uses L2 regularisation by default (``lbfgs`` solver).  For sparse
    features, ``l1`` with ``saga`` solver would be preferable but is
    omitted here for simplicity.
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10, 100]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_seed,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    param_grid = {"lr__C": C_values}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    return grid


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    logistic_cfg: dict,
    random_seed: int = 42,
) -> GridSearchCV:
    """Train Logistic Regression on training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    logistic_cfg : dict
        Logistic regression config section.
    random_seed : int
        Random seed.

    Returns
    -------
    GridSearchCV
        Fitted estimator.
    """
    model = build_logistic(
        C_values=logistic_cfg.get("C", [0.01, 0.1, 1, 10, 100]),
        max_iter=logistic_cfg.get("max_iter", 1000),
        random_seed=random_seed,
    )
    model.fit(X_train, y_train)
    logger.info(
        "LR best C=%.4f (CV F1=%.4f).",
        model.best_params_["lr__C"],
        model.best_score_,
    )
    return model


def predict_logistic(
    model: GridSearchCV,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions from a fitted Logistic Regression.

    Parameters
    ----------
    model : GridSearchCV
        Fitted estimator.
    X_test : np.ndarray
        Test features.

    Returns
    -------
    y_pred : np.ndarray
    y_prob : np.ndarray, shape (n_test, n_classes)
    """
    return model.predict(X_test), model.predict_proba(X_test)
