"""Préparation des données dérivées. N'entraîne aucun modèle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.prepare import run_preparation


def main() -> int:
    config = load_config()
    if config["phases"]["data_preparation"] != "open":
        print("Phase « data_preparation » bloquée. Aucune préparation de données lancée.")
        return 2
    run_preparation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
