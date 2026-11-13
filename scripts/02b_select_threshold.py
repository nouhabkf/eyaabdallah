"""Sélection du seuil officiel d'XGBoost. Le vendredi n'est pas ouvert."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.threshold import select_official_xgboost_threshold


def main() -> int:
    config = load_config()
    if config["phases"]["evaluation"] != "blocked":
        print("Phase « evaluation » ouverte. Le seuil n'est pas choisi tant que le test peut être lu.")
        return 2
    result = select_official_xgboost_threshold(config)
    print(
        f"Seuil XGBoost retenu : {result['selected_threshold']:.2f}. "
        "Vendredi non lu, modèles inchangés."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
