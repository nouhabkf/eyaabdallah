"""Tests du protocole figé dans la configuration."""

from __future__ import annotations

from copy import deepcopy

import pytest

from explainable_ids.config import (
    APPROVED_DATA_ROOT,
    PROJECT_ROOT,
    load_config,
    resolve_paths,
    validate_config,
)


def _set(cfg: dict, dotted: str, value: object) -> None:
    current = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def test_official_config_is_valid() -> None:
    cfg = load_config()
    assert validate_config(cfg) == []
    assert cfg["phases"]["architecture"] == "open"
    assert cfg["phases"]["data_preparation"] == "open"
    assert cfg["phases"]["training"] == "open"
    assert cfg["phases"]["evaluation"] == "open"
    assert cfg["phases"]["explanation"] == "open"
    assert cfg["phases"]["audit"] == "open"


def test_paths_keep_data_outside_onedrive() -> None:
    paths = resolve_paths(load_config())
    assert paths["data_root"].resolve() == APPROVED_DATA_ROOT.resolve()
    assert "onedrive" not in str(paths["data_root"]).lower()
    assert paths["raw"].is_relative_to(paths["data_root"])
    assert paths["stability_splits"].is_relative_to(paths["data_root"])
    assert paths["metrics"].is_relative_to(PROJECT_ROOT)
    assert paths["models"].is_relative_to(PROJECT_ROOT)
    assert "onedrive" not in str(paths["venv"]).lower()


@pytest.mark.parametrize(
    ("dotted", "value", "fragment"),
    [
        ("imbalance.smote", True, "SMOTE"),
        ("threshold.selection_split", "test", "validation"),
        ("metrics.published_pr_auc.input", "binary_predictions", "scores continus"),
        (
            "feature_decisions.allowed_evidence_splits",
            ["train", "validation", "test"],
            "train et la validation",
        ),
        ("shap_cases.selection_uses_shap_values", True, "avant"),
        ("stability.split_seed_experiments.seeds", [1001, 1001, 1003], "distincts"),
        ("compute.device", "cuda", "GPU"),
        ("paths.data_root", r"C:\Users\nouha\OneDrive\data", "OneDrive"),
        ("protocol_order", ["evaluate_test_once"], "ordre du protocole"),
        ("cleaning.order", ["impute_median"], "interdite avant"),
        ("hyperparameters.subsample.max_rows", 500000, "plafond"),
        ("threshold.always_evaluate_hundredths", [10], "0,5"),
        ("evaluation.official_test_reads", "repeated", "une fois"),
        ("dataset.download_during_architecture_phase", True, "téléchargement"),
        ("stability.pass_fail_cutoff", 0.8, "stabilité"),
        ("feature_decisions.suspect_feature.test_metrics_may_decide", True, "métrique de test"),
    ],
)
def test_protocol_violations_are_rejected(dotted: str, value: object, fragment: str) -> None:
    cfg = deepcopy(load_config())
    _set(cfg, dotted, value)
    errors = validate_config(cfg)
    assert errors
    assert any(fragment in error for error in errors)
