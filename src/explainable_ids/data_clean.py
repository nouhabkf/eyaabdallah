"""Nettoyage documenté. Responsable : personne A.

Contrat prévu
-------------
L'ordre est celui de ``cleaning.order`` dans la configuration.
Les infinis deviennent des valeurs manquantes. L'information de durée nulle
reste portée par la durée de flux. Les doublons exacts, horodatage inclus
s'il existe, sont supprimés avant le découpage.

L'imputation médiane, la suppression des colonnes constantes et la
standardisation de la régression logistique sont ajustées sur le train
seulement, dans ``preprocess``.
"""

from __future__ import annotations

from typing import Any

from explainable_ids.guards import raise_blocked


def clean_flows(config: dict[str, Any]) -> None:
    """Nettoie les flux et alimente le journal de nettoyage."""
    del config
    raise_blocked("data_preparation")
