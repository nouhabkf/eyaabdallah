"""Permutation importance du vendredi. Aucun résultat officiel n'est réécrit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.test_permutation import run_test_permutation


def main() -> int:
    result = run_test_permutation(load_config())
    top = result["top10_side_by_side"][0]["permutation_test"]
    print(
        f"Permutation terminée en {result['seconds']:.1f} s. "
        f"Première variable : {top}. "
        "Modèle officiel inchangé. Évaluation du vendredi non réécrite."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
