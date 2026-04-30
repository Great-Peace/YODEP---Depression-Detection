"""SVM classifier with RBF kernel and grid search cross-validation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_svm(
    C_values: List[float] = None,
    gamma_values: List = None,
    kernel_values: List[str] = None,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> GridSearchCV:
    """Build a grid-search SVM pipeline with standard scaling.

    Parameters
    ----------
    C_values : list of float, optional
        Regularisation values to search.  Defaults to
        ``[0.01, 0.1, 1, 10, 100]``.
    gamma_values : list, optional
        Gamma values to search.  Defaults to
        ``["scale", "auto", 0.001, 0.01]``.
    kernel_values : list of str, optional
        Kernels to search.  Defaults to ``["rbf", "linear"]``.
    cv_folds : int
        Number of inner CV folds for grid search.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    GridSearchCV
        Unfitted grid search estimator wrapping ``StandardScaler + SVC``.

    Notes
    -----
    The pipeline scales features before passing them to SVC, which is
    critical for RBF kernel performance.  Inner CV uses stratified folds
    to handle class imbalance.
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10, 100]
    if gamma_values is None:
        gamma_values = ["scale", "auto", 0.001, 0.01]
    if kernel_values is None:
        kernel_values = ["rbf", "linear"]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=random_seed, class_weight="balanced")),
        ]
    )

    param_grid = {
        "svm__C": C_values,
        "svm__gamma": gamma_values,
        "svm__kernel": kernel_values,
    }

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


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    svm_cfg: dict,
    random_seed: int = 42,
) -> GridSearchCV:
    """Train SVM with grid search on training data.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        Training features.
    y_train : np.ndarray, shape (n_train,)
        Training labels.
    svm_cfg : dict
        SVM section of ``config.yaml``.
    random_seed : int
        Random seed.

    Returns
    -------
    GridSearchCV
        Fitted estimator with best parameters found.
    """
    model = build_svm(
        C_values=svm_cfg.get("C", [0.01, 0.1, 1, 10, 100]),
        gamma_values=svm_cfg.get("gamma", ["scale", "auto", 0.001, 0.01]),
        kernel_values=svm_cfg.get("kernel", ["rbf", "linear"]),
        cv_folds=svm_cfg.get("cv_folds", 5),
        random_seed=random_seed,
    )
    model.fit(X_train, y_train)
    logger.info(
        "SVM best params: %s (CV F1=%.4f).",
        model.best_params_,
        model.best_score_,
    )
    return model


def predict_svm(
    model: GridSearchCV,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities from a fitted SVM.

    Parameters
    ----------
    model : GridSearchCV
        Fitted SVM grid search estimator.
    X_test : np.ndarray, shape (n_test, n_features)
        Test features.

    Returns
    -------
    y_pred : np.ndarray, shape (n_test,)
        Predicted class labels.
    y_prob : np.ndarray, shape (n_test, n_classes)
        Predicted class probabilities.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return y_pred, y_prob
