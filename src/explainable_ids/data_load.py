"""Chargement des CSV CIC-IDS2017. Responsable : personne A.

Contrat prévu
-------------
Entrée : CSV dans ``paths.raw``, hors OneDrive.
Sortie : parquet ``float32`` dans ``paths.interim``, plus un inventaire des fichiers.
Interdit : télécharger le dataset, inventer des lignes, calculer une statistique
de préparation sur l'ensemble du fichier avant le découpage.
"""

from __future__ import annotations

from typing import Any

from explainable_ids.guards import raise_blocked


def load_raw_csvs(config: dict[str, Any]) -> None:
    """Lit les CSV par morceaux et écrit le parquet intermédiaire."""
    del config
    raise_blocked("data_preparation")
