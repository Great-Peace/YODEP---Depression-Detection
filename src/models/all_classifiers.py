"""Generic classifier factory for all sklearn + XGBoost classifiers.

Provides :func:`build_clf_registry` which returns a dict of
``{name: (train_fn, predict_fn)}`` entries compatible with the LOSO
evaluation loop.  The three original classifiers (SVM, Logistic Regression,
Random Forest) are kept in their own modules; this module covers the rest of
sklearn's classifier suite plus XGBoost.

Classifiers included
--------------------
Tree / ensemble
  extra_trees, gradient_boosting, hist_gradient_boosting, adaboost,
  decision_tree

SVM variants
  nu_svc

Linear / discriminant
  sgd, lda, qda

Instance-based / probabilistic
  knn, naive_bayes

Neural network (sklearn)
  mlp_sklearn

External
  xgboost  (optional — skipped gracefully if package not installed)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
    logger.warning("xgboost not installed; 'xgboost' will be skipped in clf_registry.")


def build_clf_registry(
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Dict[str, Tuple[Callable, Callable]]:
    """Return a registry of all additional classifiers.

    Each entry is ``(train_fn, predict_fn)`` where:
    - ``train_fn(X_train, y_train) -> fitted GridSearchCV``
    - ``predict_fn(model, X_test) -> (y_pred, y_prob)``

    Parameters
    ----------
    cv_folds : int
        Inner stratified k-fold splits for hyperparameter search.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: classifier names; values: (train_fn, predict_fn) tuples.
    """

    def _make_train(
        clf_class,
        param_grid: dict,
        init_kwargs: Optional[dict] = None,
        has_random_state: bool = True,
        needs_scaling: bool = True,
        grid_n_jobs: int = -1,
        clf_name: str = "",
    ) -> Callable:
        def train_fn(X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
            kw = dict(init_kwargs or {})
            if has_random_state:
                kw["random_state"] = random_seed
            estimator = clf_class(**kw)
            steps = ([("scaler", StandardScaler())] if needs_scaling else []) + [("clf", estimator)]
            pipe = Pipeline(steps)
            prefixed = {f"clf__{k}": v for k, v in param_grid.items()}
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
            grid = GridSearchCV(
                pipe, prefixed, cv=cv, scoring="f1_macro",
                n_jobs=grid_n_jobs, refit=True, verbose=0,
            )
            grid.fit(X_train, y_train)
            logger.info("%s best=%s (CV F1=%.4f)", clf_name, grid.best_params_, grid.best_score_)
            return grid
        return train_fn

    def _predict_fn(model: GridSearchCV, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return model.predict(X_test), model.predict_proba(X_test)

    registry: Dict[str, Tuple[Callable, Callable]] = {}

    # ------------------------------------------------------------------ #
    # Tree / ensemble                                                      #
    # ------------------------------------------------------------------ #

    # Extra Trees — parallelise within estimator, not outer grid search
    registry["extra_trees"] = (
        _make_train(
            ExtraTreesClassifier,
            {"n_estimators": [100, 200, 500], "max_depth": [None, 5, 10, 20]},
            init_kwargs={"class_weight": "balanced", "n_jobs": -1},
            has_random_state=True, needs_scaling=False,
            grid_n_jobs=1, clf_name="ExtraTrees",
        ),
        _predict_fn,
    )

    # Gradient Boosting (sklearn's original, row-level trees)
    registry["gradient_boosting"] = (
        _make_train(
            GradientBoostingClassifier,
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7],
            },
            has_random_state=True, needs_scaling=False,
            grid_n_jobs=-1, clf_name="GradientBoosting",
        ),
        _predict_fn,
    )

    # HistGradientBoosting — sklearn's fast histogram-based GBM (LightGBM-style)
    registry["hist_gradient_boosting"] = (
        _make_train(
            HistGradientBoostingClassifier,
            {
                "max_iter": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, None],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
            init_kwargs={"class_weight": "balanced"},
            has_random_state=True, needs_scaling=False,
            grid_n_jobs=-1, clf_name="HistGradientBoosting",
        ),
        _predict_fn,
    )

    # AdaBoost — algorithm="SAMME" avoids SAMME.R deprecation warning in sklearn >=1.4
    registry["adaboost"] = (
        _make_train(
            AdaBoostClassifier,
            {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]},
            init_kwargs={"algorithm": "SAMME"},
            has_random_state=True, needs_scaling=False,
            grid_n_jobs=-1, clf_name="AdaBoost",
        ),
        _predict_fn,
    )

    # Decision Tree
    registry["decision_tree"] = (
        _make_train(
            DecisionTreeClassifier,
            {"max_depth": [None, 5, 10, 20], "criterion": ["gini", "entropy"]},
            init_kwargs={"class_weight": "balanced"},
            has_random_state=True, needs_scaling=False,
            grid_n_jobs=-1, clf_name="DecisionTree",
        ),
        _predict_fn,
    )

    # ------------------------------------------------------------------ #
    # SVM variants                                                        #
    # ------------------------------------------------------------------ #

    # Nu-SVC — alternative SVM parametrisation (nu bounds fraction of SVs)
    registry["nu_svc"] = (
        _make_train(
            NuSVC,
            {"nu": [0.1, 0.3, 0.5, 0.7], "kernel": ["rbf", "linear"]},
            init_kwargs={"probability": True, "class_weight": "balanced"},
            has_random_state=True, needs_scaling=True,
            grid_n_jobs=-1, clf_name="NuSVC",
        ),
        _predict_fn,
    )

    # ------------------------------------------------------------------ #
    # Linear / discriminant                                               #
    # ------------------------------------------------------------------ #

    # SGD Classifier — logistic regression optimised via stochastic GD
    # eta0 must be > 0 when learning_rate='adaptive'; set a safe default for both modes
    registry["sgd"] = (
        _make_train(
            SGDClassifier,
            {"alpha": [1e-4, 1e-3, 1e-2, 0.1], "learning_rate": ["optimal", "adaptive"]},
            init_kwargs={"loss": "log_loss", "class_weight": "balanced", "max_iter": 1000, "eta0": 0.01},
            has_random_state=True, needs_scaling=True,
            grid_n_jobs=-1, clf_name="SGD",
        ),
        _predict_fn,
    )

    # Linear Discriminant Analysis — solver="lsqr" required for shrinkage
    registry["lda"] = (
        _make_train(
            LinearDiscriminantAnalysis,
            {"solver": ["lsqr"], "shrinkage": [None, "auto", 0.1, 0.5]},
            has_random_state=False, needs_scaling=True,
            grid_n_jobs=-1, clf_name="LDA",
        ),
        _predict_fn,
    )

    # Quadratic Discriminant Analysis
    registry["qda"] = (
        _make_train(
            QuadraticDiscriminantAnalysis,
            {"reg_param": [0.0, 0.01, 0.1, 0.5]},
            has_random_state=False, needs_scaling=True,
            grid_n_jobs=-1, clf_name="QDA",
        ),
        _predict_fn,
    )

    # ------------------------------------------------------------------ #
    # Instance-based / probabilistic                                      #
    # ------------------------------------------------------------------ #

    # K-Nearest Neighbours
    registry["knn"] = (
        _make_train(
            KNeighborsClassifier,
            {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            has_random_state=False, needs_scaling=True,
            grid_n_jobs=-1, clf_name="KNN",
        ),
        _predict_fn,
    )

    # Gaussian Naive Bayes
    registry["naive_bayes"] = (
        _make_train(
            GaussianNB,
            {"var_smoothing": [1e-9, 1e-7, 1e-5]},
            has_random_state=False, needs_scaling=True,
            grid_n_jobs=-1, clf_name="GaussianNB",
        ),
        _predict_fn,
    )

    # ------------------------------------------------------------------ #
    # Neural network (sklearn built-in)                                   #
    # ------------------------------------------------------------------ #

    # sklearn MLP — named "mlp_sklearn" to distinguish from PyTorch MLPClassifier
    registry["mlp_sklearn"] = (
        _make_train(
            SklearnMLP,
            {
                "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [0.001, 0.01],
            },
            init_kwargs={"max_iter": 500, "early_stopping": True},
            has_random_state=True, needs_scaling=True,
            grid_n_jobs=-1, clf_name="MLP-sklearn",
        ),
        _predict_fn,
    )

    # ------------------------------------------------------------------ #
    # External: XGBoost                                                   #
    # ------------------------------------------------------------------ #

    if _HAS_XGBOOST:
        registry["xgboost"] = (
            _make_train(
                XGBClassifier,
                {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "max_depth": [3, 5, 7],
                },
                init_kwargs={"eval_metric": "logloss", "verbosity": 0, "n_jobs": 1},
                has_random_state=True, needs_scaling=False,
                grid_n_jobs=-1, clf_name="XGBoost",
            ),
            _predict_fn,
        )

    return registry
