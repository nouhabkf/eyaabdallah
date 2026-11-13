"""Les entrées non autorisées restent fermées. La stabilité n'est pas lancée."""

from __future__ import annotations

from pathlib import Path

import pytest

from explainable_ids import audit, evaluate, explain

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "func",
    [
        evaluate.select_thresholds_on_validation,
        evaluate.evaluate_test_once,
        explain.select_local_cases,
        explain.explain_local_cases,
        audit.identify_suspects_without_test,
        audit.run_stability,
    ],
)
def test_pipeline_functions_are_blocked(func) -> None:
    with pytest.raises(RuntimeError, match="non ouverte"):
        func({})


def test_audit_script_does_not_refit_or_run_stability() -> None:
    source = (ROOT / "scripts" / "05_audit.py").read_text(encoding="utf-8")
    module = (ROOT / "src" / "explainable_ids" / "feature_audit.py").read_text(encoding="utf-8")
    for text in (source, module):
        assert ".fit(" not in text
        assert "run_stability" not in text
    assert "retrain_without_suspect" not in source


def test_verify_script_does_not_train_or_read_data() -> None:
    source = (ROOT / "scripts" / "00_verify_setup.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "read_csv" not in source
    assert "urlopen" not in source
    assert "urlretrieve" not in source
