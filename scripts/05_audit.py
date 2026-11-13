"""Audit descriptif des variables. Ni ablation, ni stabilité, ni réentraînement."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.feature_audit import run_feature_audit


def main() -> int:
    config = load_config()
    if config["phases"]["audit"] != "open":
        print("Phase « audit » bloquée. Aucun audit lancé.")
        return 2
    result = run_feature_audit(config)
    print(
        f"Audit terminé. Candidats officiels : {len(result['official_candidates'])}. "
        "Aucune variable retirée."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
