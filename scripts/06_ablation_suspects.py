"""Réentraîne XGBoost sans les trois variables suspectes. Le vendredi reste fermé."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.ablation import run_ablation
from explainable_ids.config import load_config


def main() -> int:
    result = run_ablation(load_config())
    print(
        f"Ablation terminée. PR-AUC validation {result['ablated']['pr_auc']:.6f}. "
        f"Entraînement {result['ablated']['train_seconds']:.3f} s. "
        "Modèle officiel inchangé. Vendredi non scoré."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
