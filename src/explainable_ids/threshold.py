"""Seuil officiel d'XGBoost, choisi sur la validation du jeudi.

La grille, les poids et le départage viennent de la configuration figée.
Le vendredi n'est pas lu. Les modèles ne sont ni chargés ni réentraînés :
les probabilités sont celles enregistrées à la fin de l'entraînement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from explainable_ids.config import resolve_paths

EXPECTED_TRAIN_ROWS = 1_527_133
EXPECTED_VALIDATION_ROWS = 398_110
EXPECTED_VALIDATION_ATTACKS = 2_179
EXPECTED_VALIDATION_NORMAL = 395_931
LAST_SAFE_ROW_GROUP = 37
SAFE_ROWS = 1_900_000
VALIDATION_PREFIX_ROWS = SAFE_ROWS - EXPECTED_TRAIN_ROWS
VALIDATION_TAIL_ROWS = EXPECTED_VALIDATION_ROWS - VALIDATION_PREFIX_ROWS


def threshold_grid(config: dict[str, Any]) -> np.ndarray:
    """Grille figée, en centièmes. 0,50 doit y figurer."""
    grid = config["threshold"]["grid"]
    if grid.get("unit") != "hundredths":
        raise RuntimeError("La grille de seuils doit être exprimée en centièmes.")
    start = int(grid["start"])
    stop = int(grid["stop"])
    step = int(grid["step"])
    if not (1 <= start <= stop <= 99 and step >= 1):
        raise RuntimeError("La grille de seuils sort du domaine autorisé.")
    values = np.arange(start, stop + 1, step, dtype=np.float64) / 100.0
    if not np.any(np.isclose(values, 0.5)):
        raise RuntimeError("La grille de seuils ne contient pas 0,50.")
    return values


def confusion_counts(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, int]:
    """Compte avec la règle figée ``score >= seuil``."""
    predicted = scores >= threshold
    positive = labels == 1
    negative = labels == 0
    return {
        "tp": int(np.count_nonzero(predicted & positive)),
        "fn": int(np.count_nonzero(~predicted & positive)),
        "fp": int(np.count_nonzero(predicted & negative)),
        "tn": int(np.count_nonzero(~predicted & negative)),
    }


def metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    fp_weight: float,
    fn_weight: float,
    pr_auc: float,
) -> dict[str, float | int]:
    counts = confusion_counts(scores, labels, threshold)
    normal = counts["tn"] + counts["fp"]
    attack = counts["tp"] + counts["fn"]
    if normal == 0 or attack == 0:
        raise RuntimeError("Le coût exige les deux classes dans la validation.")
    fpr = counts["fp"] / normal
    fnr = counts["fn"] / attack
    predicted_positive = counts["tp"] + counts["fp"]
    precision = counts["tp"] / predicted_positive if predicted_positive else 0.0
    recall = counts["tp"] / attack
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        **counts,
        "fpr": float(fpr),
        "fnr": float(fnr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "cost": float(fp_weight * fpr + fn_weight * fnr),
    }


def select_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimise le coût, puis le FNR, puis la distance à 0,50, puis le seuil le plus bas."""

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["cost"]),
            float(row["fnr"]),
            abs(float(row["threshold"]) - 0.5),
            float(row["threshold"]),
        )

    return sorted(rows, key=sort_key)[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_thursday_labels(parquet_path: Path) -> np.ndarray:
    """Labels du jeudi. Les groupes de lignes qui touchent le vendredi ne sont pas ouverts."""
    parquet = pq.ParquetFile(parquet_path)
    table = parquet.read_row_groups(list(range(LAST_SAFE_ROW_GROUP + 1)), columns=["label"])
    if table.num_rows != SAFE_ROWS:
        raise RuntimeError("La lecture a dépassé la limite qui précède le vendredi.")
    labels = np.asarray(table.column("label").to_numpy(zero_copy_only=False), dtype=np.int8)
    validation = labels[EXPECTED_TRAIN_ROWS:]
    attacks = int(np.sum(validation == 1))
    normals = int(np.sum(validation == 0))
    if len(validation) != VALIDATION_PREFIX_ROWS or attacks != EXPECTED_VALIDATION_ATTACKS:
        raise RuntimeError("Les attaques du jeudi ne sont pas toutes dans la partie relue.")
    if normals != VALIDATION_PREFIX_ROWS - EXPECTED_VALIDATION_ATTACKS:
        raise RuntimeError("Les effectifs normaux du préfixe du jeudi sont inattendus.")
    tail = np.zeros(VALIDATION_TAIL_ROWS, dtype=np.int8)
    full = np.concatenate([validation, tail])
    if len(full) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("La validation reconstruite n'a pas 398 110 labels.")
    if int(np.sum(full == 1)) != EXPECTED_VALIDATION_ATTACKS or int(np.sum(full == 0)) != EXPECTED_VALIDATION_NORMAL:
        raise RuntimeError("Les effectifs du jeudi reconstruits ne correspondent pas au manifeste.")
    return full


def load_xgboost_validation_scores(config: dict[str, Any]) -> np.ndarray:
    paths = resolve_paths(config)
    payload = json.loads((paths["metrics"] / "training_validation.json").read_text(encoding="utf-8"))
    if payload["selected_model"] != "xgboost":
        raise RuntimeError("Le seuil officiel demandé ne correspond pas au modèle retenu.")
    recorded = next(row for row in payload["models"] if row["family"] == "xgboost")
    path = paths["metrics"] / "validation_scores_xgboost.npy"
    if _sha256_file(path) != recorded["validation_scores_sha256"]:
        raise RuntimeError("Les probabilités enregistrées d'XGBoost ont changé.")
    scores = np.load(path)
    if len(scores) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Le vecteur de probabilités n'a pas la longueur de la validation.")
    return np.asarray(scores, dtype=np.float64)


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    if config["threshold"]["score_rule"] != "score >= threshold":
        raise RuntimeError("La règle de score figée n'est pas respectée.")
    pr_auc = float(average_precision_score(labels, scores))
    fp_weight = float(config["threshold"]["fp_weight"])
    fn_weight = float(config["threshold"]["fn_weight"])
    return [
        metrics_at_threshold(
            scores,
            labels,
            float(threshold),
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            pr_auc=pr_auc,
        )
        for threshold in threshold_grid(config)
    ]


def write_cost_curve(rows: list[dict[str, Any]], chosen: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    costs = [row["cost"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds, costs, color="#1b7f4e", label="FPR + 2 × FNR")
    chosen_row = next(row for row in rows if np.isclose(row["threshold"], chosen))
    half = next(row for row in rows if np.isclose(row["threshold"], 0.5))
    axis.scatter([chosen], [chosen_row["cost"]], color="#0b3d2e", zorder=3, label=f"seuil retenu {chosen:.2f}")
    axis.scatter([0.5], [half["cost"]], color="#b85c38", zorder=3, label="seuil 0,50")
    axis.set_xlabel("Seuil")
    axis.set_ylabel("Coût")
    axis.set_title("Coût du seuil XGBoost sur la validation du jeudi")
    axis.set_xlim(0, 1)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_threshold_report(payload: dict[str, Any], path: Path) -> None:
    chosen = payload["selected"]
    half = payload["threshold_0_5"]
    lines = [
        "# Seuil officiel d'XGBoost",
        "",
        "Le seuil est choisi uniquement sur la validation du jeudi, pour le modèle déjà retenu. Les modèles n'ont pas été rechargés ni réentraînés. Les probabilités sont le vecteur enregistré à la fin de l'entraînement. Le vendredi n'a pas été lu : les groupes de lignes du parquet qui contiennent le vendredi ne sont pas ouverts.",
        "",
        "## Méthode",
        "",
        "La grille a été figée avant ce calcul : les centièmes de 0,01 à 0,99. Une attaque est prédite lorsque la probabilité est supérieure ou égale au seuil. Le coût est `FPR + 2 × FNR`, avec les poids 1 et 2 déjà verrouillés. En cas d'égalité de coût, on garde le FNR le plus bas, puis le seuil le plus proche de 0,50. La PR-AUC est calculée une fois sur les probabilités continues ; elle est la même à tous les seuils.",
        "",
        f"Les 372 867 premières lignes du jeudi ont été relues. Elles contiennent les 2 179 attaques de la validation. Les 25 243 dernières lignes du jeudi sont donc normales, par différence avec les effectifs déjà vérifiés. Leurs labels ne sont pas relus, parce qu'ils partagent un groupe de lignes avec le début du vendredi.",
        "",
        "## Seuil retenu",
        "",
        f"**{chosen['threshold']:.2f}.** Coût = {_fmt(chosen['cost'])}. "
        + (
            f"Le départage a servi : {payload['tied_costs']} seuils atteignent le coût minimal."
            if payload["tie_break_applied"]
            else "Le départage n'a pas servi : un seul seuil atteint le coût minimal."
        ),
        "",
        "Ce minimum est le bord inférieur de la grille figée. Le coût au centième suivant est plus élevé. Aucun seuil inférieur à 0,01 n'était admissible, et la grille n'a pas été étendue après le calcul."
        if abs(chosen["threshold"] - payload["sweep"][0]["threshold"]) < 1e-12
        else "Le minimum est à l'intérieur de la grille figée.",
        "",
        "| | Seuil retenu | Seuil 0,50 |",
        "|---|---:|---:|",
        f"| Coût | {_fmt(chosen['cost'])} | {_fmt(half['cost'])} |",
        f"| FPR | {_fmt(chosen['fpr'])} | {_fmt(half['fpr'])} |",
        f"| FNR | {_fmt(chosen['fnr'])} | {_fmt(half['fnr'])} |",
        f"| Précision | {_fmt(chosen['precision'])} | {_fmt(half['precision'])} |",
        f"| Rappel | {_fmt(chosen['recall'])} | {_fmt(half['recall'])} |",
        f"| F1 | {_fmt(chosen['f1'])} | {_fmt(half['f1'])} |",
        f"| TP | {chosen['tp']} | {half['tp']} |",
        f"| FP | {chosen['fp']} | {half['fp']} |",
        f"| FN | {chosen['fn']} | {half['fn']} |",
        f"| TN | {chosen['tn']} | {half['tn']} |",
        f"| PR-AUC | {_fmt(chosen['pr_auc'])} | {_fmt(half['pr_auc'])} |",
        "",
        "La courbe du coût est dans `outputs/figures/xgboost_threshold_cost.png`. Les 99 seuils sont dans `outputs/tables/xgboost_threshold_sweep.csv` et dans `outputs/metrics/xgboost_threshold.json`.",
        "",
        "## Grille complète",
        "",
        "| Seuil | TP | TN | FP | FN | FPR | FNR | Précision | Rappel | F1 | Coût |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["sweep"]:
        mark = " ←" if np.isclose(row["threshold"], chosen["threshold"]) else ""
        lines.append(
            f"| {row['threshold']:.2f}{mark} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} | "
            f"{_fmt(row['fpr'])} | {_fmt(row['fnr'])} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['f1'])} | {_fmt(row['cost'])} |"
        )
    lines.extend(
        [
            "",
            "L'évaluation du vendredi n'est pas faite. Elle attend une autorisation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def select_official_xgboost_threshold(config: dict[str, Any]) -> dict[str, Any]:
    """Choisit et enregistre le seuil d'XGBoost. N'ouvre pas le vendredi et n'ajuste aucun modèle."""
    if config["phases"]["evaluation"] != "blocked":
        raise RuntimeError("L'évaluation finale n'est pas autorisée dans cette étape.")
    if config["threshold"]["selection_split"] != "validation":
        raise RuntimeError("Le seuil se choisit sur la validation.")
    if "test" not in config["threshold"]["forbidden_splits"]:
        raise RuntimeError("Le test est interdit pour le choix du seuil.")
    scores = load_xgboost_validation_scores(config)
    labels = load_thursday_labels(resolve_paths(config)["interim"] / "flows.parquet")
    rows = sweep_thresholds(scores, labels, config)
    chosen = select_threshold(rows)
    tied = [row for row in rows if row["cost"] == chosen["cost"]]
    half = next(row for row in rows if np.isclose(row["threshold"], 0.5))
    paths = resolve_paths(config)
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    figure = paths["figures"] / "xgboost_threshold_cost.png"
    table = paths["tables"] / "xgboost_threshold_sweep.csv"
    result = paths["metrics"] / "xgboost_threshold.json"
    write_cost_curve(rows, float(chosen["threshold"]), figure)
    _write_table(table, rows, float(chosen["threshold"]))
    payload = {
        "model": "xgboost",
        "split": "validation",
        "friday_read": False,
        "models_modified": False,
        "models_refit": False,
        "score_source": "outputs/metrics/validation_scores_xgboost.npy",
        "score_rule": "score >= threshold",
        "objective": "FPR + 2 * FNR",
        "grid": "hundredths 0.01 to 0.99",
        "selected_threshold": chosen["threshold"],
        "tie_break_applied": len(tied) > 1,
        "tied_costs": len(tied),
        "tie_break": ["lower_fnr", "closer_to_0_5", "lower_threshold"],
        "selected": {key: value for key, value in chosen.items() if key != "tie_break_applied"},
        "threshold_0_5": half,
        "pr_auc": rows[0]["pr_auc"],
        "sweep": rows,
        "figure": str(figure),
        "table": str(table),
    }
    temporary = result.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(result)
    report = paths["project_root"] / "docs" / "threshold_selection.md"
    write_threshold_report(payload, report)
    return payload


def _write_table(path: Path, rows: list[dict[str, Any]], chosen: float) -> None:
    header = ["threshold", "selected", "tp", "tn", "fp", "fn", "fpr", "fnr", "precision", "recall", "f1", "cost", "pr_auc"]
    lines = [",".join(header)]
    for row in rows:
        values = [
            f"{row['threshold']:.2f}",
            "true" if np.isclose(row["threshold"], chosen) else "false",
            str(row["tp"]),
            str(row["tn"]),
            str(row["fp"]),
            str(row["fn"]),
            repr(row["fpr"]),
            repr(row["fnr"]),
            repr(row["precision"]),
            repr(row["recall"]),
            repr(row["f1"]),
            repr(row["cost"]),
            repr(row["pr_auc"]),
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
