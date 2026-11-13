"""Stabilité SHAP. Le vendredi et le modèle officiel ne sont pas touchés."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.stability_explain import run_stability_explanations


def main() -> int:
    result = run_stability_explanations(load_config())
    print(
        "Stabilité terminée. "
        f"Jaccard A {result['pairs_a'][0]['jaccard_top10']:.3f}. "
        "Modèle officiel inchangé. Vendredi non lu."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
