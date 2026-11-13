"""Découpages officiels, de stabilité et chronologique. Responsable : personne A.

Contrat prévu
-------------
Le découpage officiel est chronologique : lundi–mercredi, jeudi, vendredi.
Le découpage groupé 70 / 15 / 15, graine 42, est le complément. Il ne choisit
ni le modèle officiel ni le seuil officiel.

Les trois graines de découpage de stabilité produisent trois partitions
neuves, stratifiées, documentées par effectifs et empreintes SHA-256.
Elles ne recopient pas les indices officiels.

Les trois graines de modèle réutilisent la partition officielle : seule la
graine d'algorithme change.

Le découpage chronologique n'est produit que si un horodatage ou un ordre
de fichiers exploitable existe. Il ne choisit aucun réglage.
"""

from __future__ import annotations

from typing import Any

from explainable_ids.guards import raise_blocked


def write_official_split(config: dict[str, Any]) -> None:
    """Écrit les indices train, validation et test du découpage principal."""
    del config
    raise_blocked("data_preparation")


def write_stability_partitions(config: dict[str, Any]) -> None:
    """Écrit les partitions indépendantes et les répétitions de graine modèle."""
    del config
    raise_blocked("audit")


def write_chronological_partition_if_possible(config: dict[str, Any]) -> None:
    """Écrit le complément chronologique, ou la raison de son absence."""
    del config
    raise_blocked("data_preparation")
