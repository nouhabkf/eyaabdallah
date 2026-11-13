"""Gardes des phases encore fermées."""

from __future__ import annotations


def raise_blocked(phase: str) -> None:
    """Empêche l'exécution d'une étape dont l'implémentation n'est pas ouverte."""
    raise RuntimeError(
        f"Phase « {phase} » non ouverte. "
        "Cette étape reste fermée tant qu'elle n'est pas autorisée."
    )


def blocked_script(phase: str, blocked_message: str) -> int:
    """Point d'entrée des scripts. Retourne 2 tant que l'étape n'est pas livrée."""
    from explainable_ids.config import load_config

    config = load_config()
    state = config["phases"][phase]
    if state != "open":
        print(f"Phase « {phase} » bloquée ({state}). {blocked_message}")
        return 2
    print(
        f"Phase « {phase} » est marquée ouverte, "
        "mais cette livraison n'implémente pas encore cette étape."
    )
    return 2
