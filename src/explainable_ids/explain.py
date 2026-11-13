"""Importances et SHAP. Responsables : personne A et personne B.

Le SHAP autorisé est l'illustration du XGBoost figé sur le vendredi.
L'impureté, la permutation et le SHAP de validation restent dans l'audit,
qui n'est pas ouvert. Les cinq cas suivent ``scores_and_labels_only_v1``
avant leurs graphiques.
"""

from __future__ import annotations

from typing import Any

from explainable_ids.guards import raise_blocked


def rf_impurity(config: dict[str, Any]) -> None:
    """Exporte l'importance d'impureté de la forêt aléatoire."""
    del config
    raise_blocked("audit")


def permutation_importance_on_validation(config: dict[str, Any]) -> None:
    """Calcule l'importance par permutation sur la validation."""
    del config
    raise_blocked("audit")


def shap_global_on_validation(config: dict[str, Any]) -> None:
    """Le SHAP qui servirait à une décision de variable reste fermé."""
    del config
    raise_blocked("audit")


def select_local_cases(config: dict[str, Any]) -> None:
    """Réservé au script d'explication, qui choisit les cas avant SHAP."""
    if not isinstance(config, dict) or config.get("phases", {}).get("explanation") != "open":
        raise_blocked("explanation")
    raise RuntimeError("Les cas sont choisis dans run_shap_explanation, avant les graphiques.")


def explain_local_cases(config: dict[str, Any]) -> None:
    """Réservé au script d'explication, après la sélection des cas."""
    if not isinstance(config, dict) or config.get("phases", {}).get("explanation") != "open":
        raise_blocked("explanation")
    raise RuntimeError("Les graphiques locaux sont produits par run_shap_explanation.")
