"""Rapport alimenté par les fichiers de métriques. Responsables : les deux.

Contrat prévu
-------------
Aucun chiffre du rapport n'est saisi à la main. Les tables et les figures
sont produites depuis ``outputs/metrics``.
"""

from __future__ import annotations

from typing import Any

from explainable_ids.guards import raise_blocked


def collect_metric_files(config: dict[str, Any]) -> None:
    """Assemble les métriques déjà écrites par les scripts d'évaluation."""
    del config
    raise_blocked("evaluation")
