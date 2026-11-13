"""Mesures de stabilité, sans lire les données."""

from __future__ import annotations

from pathlib import Path

from explainable_ids.stability_explain import jaccard_top, spearman_ranks

ROOT = Path(__file__).resolve().parents[1]


def test_jaccard_of_identical_tops_is_one() -> None:
    names = ["A", "B", "C"]
    assert jaccard_top(names, names) == 1.0


def test_jaccard_of_disjoint_tops_is_zero() -> None:
    assert jaccard_top(["A", "B"], ["C", "D"]) == 0.0


def test_spearman_detects_a_reversal() -> None:
    forward = {"A": 1, "B": 2, "C": 3}
    backward = {"A": 3, "B": 2, "C": 1}
    assert spearman_ranks(forward, forward) == 1.0
    assert spearman_ranks(forward, backward) == -1.0


def test_stability_source_does_not_replace_the_official_model() -> None:
    source = (ROOT / "src" / "explainable_ids" / "stability_explain.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "07_stability_explanations.py").read_text(encoding="utf-8")
    assert "xgboost.joblib" not in script
    assert "friday_evaluation.json" not in source or "sha256" in source
    assert "test.npy" not in source
