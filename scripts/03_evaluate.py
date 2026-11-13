"""Évaluation finale du vendredi. SHAP et l'audit ne sont pas lancés."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.evaluate import evaluate_test_once


def main() -> int:
    config = load_config()
    if config["phases"]["evaluation"] != "open":
        print("Phase « evaluation » bloquée. Aucune évaluation lancée.")
        return 2
    result = evaluate_test_once(config)
    print(
        f"Vendredi évalué. PR-AUC {result['pr_auc']:.6f}. "
        "Modèle et seuil inchangés."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
