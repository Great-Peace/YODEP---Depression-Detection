"""Random Forest classifier with grid search and feature importance extraction."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_random_forest(
    n_estimators_values: List[int] = None,
    max_depth_values: List[Optional[int]] = None,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> GridSearchCV:
    """Build a grid-search Random Forest pipeline.

    Parameters
    ----------
    n_estimators_values : list of int, optional
        Ensemble sizes to search.  Defaults to ``[100, 200, 500]``.
    max_depth_values : list of int or None, optional
        Tree depths to search.  Defaults to ``[None, 5, 10, 20]``.
    cv_folds : int
        Inner CV folds.
    random_seed : int
        Random seed.

    Returns
    -------
    GridSearchCV
        Unfitted estimator.

    Notes
    -----
    Random Forest does not require scaling, but the Pipeline wrapper is
    included for API consistency with other models.
    """
    if n_estimators_values is None:
        n_estimators_values = [100, 200, 500]
    if max_depth_values is None:
        max_depth_values = [None, 5, 10, 20]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    random_state=random_seed,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "rf__n_estimators": n_estimators_values,
        "rf__max_depth": max_depth_values,
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,  # RF is already parallelised; avoid nested parallelism
        refit=True,
        verbose=0,
    )
    return grid


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_cfg: dict,
    random_seed: int = 42,
) -> GridSearchCV:
    """Train Random Forest on training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    rf_cfg : dict
        Random forest config section.
    random_seed : int
        Random seed.

    Returns
    -------
    GridSearchCV
        Fitted estimator.
    """
    model = build_random_forest(
        n_estimators_values=rf_cfg.get("n_estimators", [100, 200, 500]),
        max_depth_values=rf_cfg.get("max_depth", [None, 5, 10, 20]),
        cv_folds=rf_cfg.get("cv_folds", 5),
        random_seed=random_seed,
    )
    model.fit(X_train, y_train)
    logger.info(
        "RF best params: %s (CV F1=%.4f).",
        model.best_params_,
        model.best_score_,
    )
    return model


def predict_random_forest(
    model: GridSearchCV,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions from a fitted Random Forest.

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


def get_feature_importances(
    model: GridSearchCV,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract feature importances from the fitted Random Forest.

    Parameters
    ----------
    model : GridSearchCV
        Fitted grid search estimator.
    feature_names : list of str
        Feature names corresponding to the training feature matrix columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature_name``, ``importance``.  Sorted descending.

    Notes
    -----
    Used by Extra 4 (Random Forest feature importance).  If F0 features
    dominate in English but not Yoruba, that is direct evidence for the
    Acoustic-Linguistic Confound hypothesis.
    """
    rf = model.best_estimator_.named_steps["rf"]
    importances = rf.feature_importances_

    if len(importances) != len(feature_names):
        logger.warning(
            "Feature importance length (%d) != feature_names length (%d). "
            "Names will be truncated/padded.",
            len(importances),
            len(feature_names),
        )
        min_len = min(len(importances), len(feature_names))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    df = pd.DataFrame(
        {"feature_name": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return df
