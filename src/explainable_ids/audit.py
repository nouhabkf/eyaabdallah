"""Audit des variables. La stabilité et l'ablation restent fermées.

Le classement officiel croise l'impureté de la forêt, la permutation sur
la validation et le SHAP sur la validation. Le test ne désigne pas une
variable suspecte. Aucune variable n'est retirée à cette étape.
"""

from __future__ import annotations

from typing import Any

from explainable_ids.guards import raise_blocked


def rank_top_features(config: dict[str, Any]) -> None:
    """Le classement est produit par le script d'audit, sans ablation."""
    if not isinstance(config, dict) or config.get("phases", {}).get("audit") != "open":
        raise_blocked("audit")
    raise RuntimeError("Le classement s'obtient par run_feature_audit.")


def identify_suspects_without_test(config: dict[str, Any]) -> None:
    """Applique les critères d'audit sans lire les métriques de test."""
    if not isinstance(config, dict) or config.get("phases", {}).get("audit") != "open":
        raise_blocked("audit")
    raise RuntimeError("Les critères s'appliquent dans run_feature_audit.")


def retrain_without_suspect(config: dict[str, Any]) -> None:
    """Aucune ablation n'est autorisée tant que l'expérience n'est pas demandée."""
    del config
    raise RuntimeError("Aucune variable n'est retirée et aucun modèle n'est réentraîné.")


def run_stability(config: dict[str, Any]) -> None:
    """L'analyse de stabilité n'est pas autorisée dans cet audit."""
    if not isinstance(config, dict) or config.get("phases", {}).get("audit") != "open":
        raise_blocked("audit")
    raise RuntimeError("L'analyse de stabilité n'est pas autorisée.")
