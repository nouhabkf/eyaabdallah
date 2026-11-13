"""Contrats de l'évaluation finale, sans relire le vendredi."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from explainable_ids.friday_eval import _metrics

ROOT = Path(__file__).resolve().parents[1]


def test_two_thresholds_share_the_same_continuous_score() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.2, 0.4, 0.9])
    low = _metrics(labels, scores, 0.01)
    half = _metrics(labels, scores, 0.50)
    assert low["fp"] == 1
    assert low["fn"] == 0
    assert half["fp"] == 0
    assert half["fn"] == 1
    assert low["accuracy"] == 0.75
    assert half["fpr"] == 0.0
    assert low["fnr"] == 0.0


def test_evaluation_source_does_not_refit_or_explain() -> None:
    source = (ROOT / "src" / "explainable_ids" / "friday_eval.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "03_evaluate.py").read_text(encoding="utf-8")
    for text in (source, script):
        assert ".fit(" not in text
        assert "import shap" not in text
        assert "TreeExplainer" not in text
        assert "threshold_grid" not in text
