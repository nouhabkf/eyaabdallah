"""Règle de sélection des cas SHAP, sans lire le vendredi."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from explainable_ids.shap_run import representative_indices, select_score_cases

ROOT = Path(__file__).resolve().parents[1]


def test_case_rule_picks_scores_before_any_explanation() -> None:
    labels = np.array([0, 1, 1, 0, 1])
    scores = np.array([0.20, 0.90, 0.011, 0.015, 0.009])
    chosen = {row["id"]: row for row in select_score_cases(labels, scores, 0.01)}
    assert chosen["tp_confident"]["index"] == 1
    assert chosen["tp_marginal"]["index"] == 2
    assert chosen["fp_confident"]["index"] == 0
    assert chosen["fp_marginal"]["index"] == 3
    assert chosen["fn_marginal"]["index"] == 4


def test_exact_score_tie_keeps_the_smallest_index() -> None:
    labels = np.array([1, 1, 0])
    scores = np.array([0.80, 0.80, 0.00])
    chosen = select_score_cases(labels, scores, 0.01)[0]
    assert chosen["id"] == "tp_confident"
    assert chosen["index"] == 0
    assert chosen["n_tied"] == 2


def test_missing_category_is_not_replaced() -> None:
    labels = np.array([1, 1])
    scores = np.array([0.40, 0.20])
    chosen = {row["id"]: row for row in select_score_cases(labels, scores, 0.01)}
    assert chosen["fp_confident"]["present"] is False
    assert chosen["fp_confident"]["index"] is None
    assert chosen["fn_marginal"]["present"] is False
    assert chosen["tp_confident"]["present"] is True


def test_sample_follows_group_proportions_and_stays_inside_the_cap() -> None:
    groups = np.array(["A"] * 70 + ["B"] * 30)
    indices, allocation = representative_indices(groups, 10, 42)
    assert len(indices) == 10
    assert indices.tolist() == sorted(indices.tolist())
    counts = {row["label"]: row["sampled"] for row in allocation}
    assert counts == {"A": 7, "B": 3}
    assert set(indices).issubset(set(range(100)))


def test_shap_runner_does_not_refit_or_audit() -> None:
    source = (ROOT / "src" / "explainable_ids" / "shap_run.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "04_explain.py").read_text(encoding="utf-8")
    for text in (source, script):
        assert ".fit(" not in text
        assert "KernelExplainer" not in text
        assert "permutation_importance" not in text
        assert "identify_suspects" not in text
        assert "friday_evaluation.json" not in text or "read_text" in text
    assert "friday_evaluation.json" not in script
