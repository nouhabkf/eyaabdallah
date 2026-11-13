"""Entraînement chronologique. Le test, SHAP et l'audit ne sont pas lancés."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config
from explainable_ids.train import run_official_training


def main() -> int:
    config = load_config()
    if config["phases"]["training"] != "open":
        print("Phase « training » bloquée. Aucun entraînement lancé.")
        return 2
    run_official_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
