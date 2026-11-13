"""SHAP du XGBoost figé. L'audit des variables n'est pas lancé."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.shap_run import run_shap_explanation


def main() -> int:
    config = load_config()
    if config["phases"]["explanation"] != "open":
        print("Phase « explanation » bloquée. Aucune explication lancée.")
        return 2
    if config["phases"]["audit"] != "blocked":
        print("Phase « audit » non autorisée. Aucune explication lancée.")
        return 2
    result = run_shap_explanation(config)
    top = result["top_10"][0]["feature"]
    print(
        f"SHAP terminé. Première variable : {top}. "
        f"Écart de marge {result['coherence']['max_abs_margin_error']:.3e}. "
        "Modèle et seuil inchangés."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
