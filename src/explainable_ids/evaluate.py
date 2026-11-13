"""Sélection du seuil sur la validation, puis lecture unique du test.

Responsable : personne B.

Contrat prévu
-------------
Le score est ``predict_proba`` de la classe attaque.
La PR-AUC publiée est ``average_precision_score(y_true, score)``.
Elle n'est jamais calculée à partir des prédictions binaires.

Pour chaque modèle, le seuil retenu minimise sur la validation
``fp_weight * FPR + fn_weight * FNR``, avec les poids figés dans la
configuration. Le seuil 0,5 est aussi évalué.
Le modèle principal est celui dont la PR-AUC de validation est la plus haute.

Le test est lu une fois, aux deux seuils, pour les trois modèles.
"""

from __future__ import annotations

from typing import Any

from explainable_ids.guards import raise_blocked


def select_primary_model_on_validation(config: dict[str, Any]) -> None:
    """Choisit le modèle principal par PR-AUC de validation."""
    del config
    raise_blocked("evaluation")


def select_thresholds_on_validation(config: dict[str, Any]) -> None:
    """Choisit un seuil par modèle, exclusivement sur la validation."""
    del config
    raise_blocked("evaluation")


def evaluate_test_once(config: dict[str, Any]) -> dict[str, Any]:
    """Évalue le vendredi une seule fois, aux seuils déjà figés."""
    phase = config.get("phases", {}).get("evaluation") if isinstance(config, dict) else None
    if phase != "open":
        raise_blocked("evaluation")
    from explainable_ids.friday_eval import run_friday_evaluation

    return run_friday_evaluation(config)
