"""Pipelines préparés, non ajustés sur CIC-IDS2017."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from explainable_ids.config import load_config
from explainable_ids.preprocess import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    build_tree_pipeline,
    build_xgboost_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]


def test_three_pipelines_are_unfitted_and_trees_are_not_scaled() -> None:
    config = load_config()
    logistic = build_logistic_pipeline(config)
    forest = build_random_forest_pipeline(config)
    boosted = build_xgboost_pipeline(config)
    assert list(logistic.named_steps) == ["imputer", "scaler", "classifier"]
    assert list(forest.named_steps) == ["imputer", "classifier"]
    assert list(boosted.named_steps) == ["imputer", "classifier"]
    assert logistic.named_steps["imputer"].strategy == "median"
    assert forest.named_steps["classifier"].class_weight == "balanced"
    assert boosted.named_steps["classifier"].get_params()["device"] == "cpu"
    assert boosted.named_steps["classifier"].get_params()["tree_method"] == "hist"
    assert boosted.named_steps["classifier"].get_params()["scale_pos_weight"] is None
    for pipeline in (logistic, forest, boosted):
        with pytest.raises(Exception):
            check_is_fitted(pipeline.named_steps["imputer"])
    assert build_tree_pipeline(config, "xgboost").named_steps["classifier"].get_params()["device"] == "cpu"


def test_learned_scaler_uses_only_the_rows_passed_as_training() -> None:
    config = load_config()
    train = np.array([[1.0, 10.0], [np.nan, 30.0], [3.0, 50.0], [5.0, np.nan]], dtype=float)
    held_out = np.array([[1000.0, -1000.0]], dtype=float)
    learned = build_logistic_pipeline(config)[:-1]
    learned.fit(train)
    filled = train.copy()
    for column in range(train.shape[1]):
        median = np.nanmedian(train[:, column])
        missing = np.isnan(filled[:, column])
        filled[missing, column] = median
    assert learned.named_steps["scaler"].mean_ == pytest.approx(filled.mean(axis=0))
    contaminated = np.vstack([filled, held_out])
    assert learned.named_steps["scaler"].mean_ != pytest.approx(contaminated.mean(axis=0))


def test_preparation_source_does_not_fit_a_model() -> None:
    source = (ROOT / "src" / "explainable_ids" / "prepare.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "01_prepare.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert ".fit(" not in script
    assert "import shap" not in source
    assert "TreeExplainer" not in source
    assert 'open("wb"' not in source
    assert "raw" in source
