"""Règle de seuil, sans données du vendredi et sans modèle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from explainable_ids.config import load_config
from explainable_ids.threshold import (
    confusion_counts,
    metrics_at_threshold,
    select_threshold,
    threshold_grid,
)

ROOT = Path(__file__).resolve().parents[1]


def test_grid_is_the_locked_hundredths_and_contains_one_half() -> None:
    grid = threshold_grid(load_config())
    assert grid[0] == 0.01
    assert grid[-1] == 0.99
    assert len(grid) == 99
    assert np.any(np.isclose(grid, 0.5))


def test_positive_prediction_uses_greater_or_equal() -> None:
    scores = np.array([0.5, 0.49])
    labels = np.array([1, 0])
    counts = confusion_counts(scores, labels, 0.5)
    assert counts == {"tp": 1, "fn": 0, "fp": 0, "tn": 1}


def test_cost_is_fpr_plus_twice_fnr_and_lower_fnr_breaks_ties() -> None:
    scores = np.array([0.2, 0.9, 0.4, 0.8])
    labels = np.array([0, 1, 1, 0])
    row = metrics_at_threshold(scores, labels, 0.5, fp_weight=1, fn_weight=2, pr_auc=0.5)
    assert row["fp"] == 1
    assert row["fn"] == 1
    assert row["cost"] == 1 * 0.5 + 2 * 0.5
    chosen = select_threshold(
        [
            {"threshold": 0.20, "cost": 0.4, "fnr": 0.2},
            {"threshold": 0.80, "cost": 0.4, "fnr": 0.1},
            {"threshold": 0.40, "cost": 0.5, "fnr": 0.0},
        ]
    )
    assert chosen["threshold"] == 0.80


def test_threshold_source_does_not_open_friday_or_refit() -> None:
    source = (ROOT / "src" / "explainable_ids" / "threshold.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "02b_select_threshold.py").read_text(encoding="utf-8")
    for text in (source, script):
        assert "test.npy" not in text
        assert "Friday-" not in text
        assert ".fit(" not in text
        assert "import shap" not in text
        assert "joblib" not in text
