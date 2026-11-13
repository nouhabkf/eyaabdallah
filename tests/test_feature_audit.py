"""Règle officielle de candidat, sans lire les données."""

from __future__ import annotations

import pytest

from explainable_ids.feature_audit import official_candidates


def test_two_methods_are_required() -> None:
    rankings = {
        "rf_impurity": ["A", "B"],
        "permutation_validation": ["A", "C"],
        "shap_validation": ["D", "B"],
    }
    chosen = official_candidates(rankings, top_k=2, minimum_methods=2)
    assert [row["feature"] for row in chosen] == ["A", "B"]
    assert all(row["n_methods"] == 2 for row in chosen)


def test_a_single_method_does_not_qualify() -> None:
    rankings = {
        "rf_impurity": ["A"],
        "permutation_validation": ["B"],
        "shap_validation": ["C"],
    }
    assert official_candidates(rankings, top_k=1, minimum_methods=2) == []


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        official_candidates({"rf_impurity": ["A"], "test_shap": ["A"]}, top_k=1, minimum_methods=2)
