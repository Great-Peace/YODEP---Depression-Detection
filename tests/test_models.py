"""Tests for classifier models using small synthetic datasets."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification


def _binary_data(n_samples=60, n_features=20, seed=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


def test_svm_train_predict():
    from src.models.svm_model import train_svm, predict_svm

    X, y = _binary_data()
    cfg = {"C": [0.1, 1], "gamma": ["scale"], "kernel": ["rbf"], "cv_folds": 2}
    model = train_svm(X, y, cfg, random_seed=42)
    y_pred, y_prob = predict_svm(model, X)

    assert len(y_pred) == len(y)
    assert y_prob.shape == (len(y), 2)
    assert set(y_pred).issubset({0, 1})


def test_logistic_train_predict():
    from src.models.logistic_model import train_logistic, predict_logistic

    X, y = _binary_data()
    cfg = {"C": [0.1, 1], "max_iter": 200}
    model = train_logistic(X, y, cfg, random_seed=42)
    y_pred, y_prob = predict_logistic(model, X)

    assert len(y_pred) == len(y)
    assert y_prob.shape == (len(y), 2)


def test_random_forest_train_predict():
    from src.models.random_forest import train_random_forest, predict_random_forest

    X, y = _binary_data()
    cfg = {"n_estimators": [50], "max_depth": [5], "cv_folds": 2}
    model = train_random_forest(X, y, cfg, random_seed=42)
    y_pred, y_prob = predict_random_forest(model, X)

    assert len(y_pred) == len(y)
    assert y_prob.shape == (len(y), 2)


def test_random_forest_feature_importances():
    from src.models.random_forest import (
        train_random_forest,
        get_feature_importances,
    )

    X, y = _binary_data(n_features=10)
    cfg = {"n_estimators": [50], "max_depth": [5], "cv_folds": 2}
    names = [f"feat_{i}" for i in range(10)]
    model = train_random_forest(X, y, cfg, random_seed=42)
    imp_df = get_feature_importances(model, names)

    assert len(imp_df) == 10
    assert "feature_name" in imp_df.columns
    assert "importance" in imp_df.columns
    # Should be sorted descending
    assert imp_df["importance"].is_monotonic_decreasing


def test_mlp_train_predict():
    from src.models.mlp_fusion import MLPClassifier

    X, y = _binary_data(n_samples=80, n_features=32)
    X_tr, X_val = X[:60], X[60:]
    y_tr, y_val = y[:60], y[60:]

    mlp = MLPClassifier(
        input_dim=32,
        hidden_dims=[32, 16],
        dropout=0.1,
        epochs=5,
        batch_size=16,
        random_seed=42,
    )
    mlp.fit(X_tr, y_tr, X_val, y_val)
    y_pred, y_prob = mlp.predict(X_val)

    assert len(y_pred) == len(y_val)
    assert y_prob.shape == (len(y_val), 2)
    assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-5)


def test_metrics_compute():
    from src.evaluation.metrics import compute_metrics

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_prob = np.array([[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]])

    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_bootstrap_ci_basic():
    from src.evaluation.metrics import bootstrap_ci

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0, 0])

    lo, hi = bootstrap_ci(y_true, y_pred, n_iterations=50, random_seed=42)
    assert lo <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
