"""Contrats d'entraînement sans ajuster les modèles sur CIC-IDS2017."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from explainable_ids.config import load_config
from explainable_ids.train import (
    PositiveWeightXGBClassifier,
    make_training_pipeline,
    search_hyperparameters,
    select_primary,
    stratified_subsample_indices,
    train_column_mask,
)

ROOT = Path(__file__).resolve().parents[1]


def test_column_mask_drops_constants_and_exact_duplicates() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 1.0, 5.0],
            [2.0, 0.0, 2.0, 4.0],
            [3.0, 0.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    kept, constant, duplicate = train_column_mask(matrix)
    assert constant == [1]
    assert duplicate == [2]
    assert kept.tolist() == [0, 3]


def test_subsample_is_stratified_reproducible_and_capped() -> None:
    labels = np.array([0] * 80 + [1] * 20)
    first = stratified_subsample_indices(labels, 40, 42)
    second = stratified_subsample_indices(labels, 40, 42)
    assert len(first) == 40
    assert np.array_equal(first, second)
    assert int(np.sum(labels[first] == 1)) == 8
    assert np.all(first[1:] > first[:-1])


def test_selection_uses_validation_pr_auc_then_recall_then_time() -> None:
    order = ["logistic_regression", "random_forest", "xgboost"]
    rows = [
        {"family": "logistic_regression", "average_precision": 0.40, "recall_at_0_5": 0.9, "train_seconds": 1},
        {"family": "random_forest", "average_precision": 0.50, "recall_at_0_5": 0.2, "train_seconds": 50},
        {"family": "xgboost", "average_precision": 0.50, "recall_at_0_5": 0.2, "train_seconds": 10},
    ]
    assert select_primary(rows, order) == "xgboost"
    rows[0]["average_precision"] = 0.50
    rows[0]["recall_at_0_5"] = 0.95
    assert select_primary(rows, order) == "logistic_regression"


def test_scale_pos_weight_is_computed_from_the_fitted_labels() -> None:
    features = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    first = np.array([0, 0, 0, 0, 1, 1])
    model = PositiveWeightXGBClassifier(
        n_estimators=2,
        max_depth=1,
        tree_method="hist",
        device="cpu",
        objective="binary:logistic",
        random_state=42,
        verbosity=0,
    )
    model.fit(features, first)
    assert model.get_params()["scale_pos_weight"] == pytest.approx(4 / 2)
    model.fit(features[:4], np.array([0, 0, 0, 1]))
    assert model.get_params()["scale_pos_weight"] == pytest.approx(3 / 1)


def test_inner_search_uses_continuous_scores_and_does_not_refit() -> None:
    config = deepcopy(load_config())
    config["hyperparameters"]["inner_cv"]["folds"] = 2
    config["hyperparameters"]["logistic_regression"]["param_grid"] = {"C": [0.1, 1.0]}
    generator = np.random.default_rng(42)
    features = generator.normal(size=(40, 3)).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20)
    features[:20] -= 1.0
    features[20:] += 1.0
    search = search_hyperparameters(
        make_training_pipeline(config, "logistic_regression"),
        "logistic_regression",
        config,
        features,
        labels,
    )
    assert set(search["best_params"]) == {"C"}
    assert search["inner_cv_average_precision"] > 0.5
    assert search["refit_on_search_sample"] is False


def test_training_sources_do_not_score_the_test_or_call_shap() -> None:
    source = (ROOT / "src" / "explainable_ids" / "train.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "02_train.py").read_text(encoding="utf-8")
    for text in (source, script):
        assert "import shap" not in text
        assert "TreeExplainer" not in text
        assert "test.npy" not in text
        assert "evaluate_test_once" not in text
