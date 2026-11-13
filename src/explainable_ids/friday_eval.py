"""Évaluation finale du vendredi avec le XGBoost déjà entraîné.

Les seuils 0,50 et 0,01 sont lus, pas recalculés. Aucun paramètre n'est
ajusté à partir de ces résultats. SHAP n'est pas appelé.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from joblib import load
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from explainable_ids.config import resolve_paths

FRIDAY_START = 1_925_243
EXPECTED_TEST_ROWS = 595_723
EXPECTED_TEST_NORMAL = 375_502
EXPECTED_TEST_ATTACK = 220_221
REFERENCE_THRESHOLD = 0.50
OFFICIAL_THRESHOLD = 0.01


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    total = int(tn + fp + fn + tp)
    normal = int(tn + fp)
    attack = int(tp + fn)
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tn + tp) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / normal),
        "fnr": float(fn / attack),
    }


def _family_table(names: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    predicted = scores >= threshold
    rows = []
    for name in sorted(set(names.tolist())):
        mask = names == name
        positive = labels[mask] == 1
        flagged = predicted[mask]
        rows.append(
            {
                "label": str(name),
                "rows": int(mask.sum()),
                "tp": int(np.count_nonzero(flagged & positive)),
                "fn": int(np.count_nonzero(~flagged & positive)),
                "fp": int(np.count_nonzero(flagged & ~positive)),
                "tn": int(np.count_nonzero(~flagged & ~positive)),
            }
        )
    return rows


def _load_friday(feature_names: list[str], parquet_path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(parquet_path)
    group_rows = [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)]
    offset = 0
    groups = []
    for index, count in enumerate(group_rows):
        if offset + count > FRIDAY_START:
            groups.append(index)
        offset += count
    table = parquet.read_row_groups(groups, columns=feature_names + ["label", "label_original"])
    seen = sum(group_rows[: groups[0]])
    skip = FRIDAY_START - seen
    if skip < 0:
        raise RuntimeError("Le premier groupe lu commence après le vendredi.")
    labels = np.asarray(table.column("label").to_numpy(zero_copy_only=False)[skip:], dtype=np.int8)
    originals = np.asarray(table.column("label_original").to_numpy(zero_copy_only=False)[skip:], dtype=object)
    matrix = np.empty((len(labels), len(feature_names)), dtype=np.float32)
    for index, name in enumerate(feature_names):
        matrix[:, index] = table.column(name).to_numpy(zero_copy_only=False)[skip:]
    if len(labels) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Le vendredi relu a {len(labels)} lignes, pas {EXPECTED_TEST_ROWS}.")
    if int(np.sum(labels == 0)) != EXPECTED_TEST_NORMAL or int(np.sum(labels == 1)) != EXPECTED_TEST_ATTACK:
        raise RuntimeError("Les effectifs Normal/Attack du vendredi ne correspondent pas au manifeste.")
    return {"features": matrix, "labels": labels, "originals": originals}


def _time_predictions(model, features: np.ndarray, repeats: int) -> tuple[list[float], np.ndarray]:
    times = []
    scores = None
    for _ in range(repeats):
        started = time.perf_counter()
        scores = model.predict_proba(features)[:, 1]
        times.append(time.perf_counter() - started)
    return times, np.asarray(scores, dtype=np.float64)


def _plots(labels: np.ndarray, scores: np.ndarray, rows: list[dict[str, Any]], paths: dict[str, Path]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths["figures"].mkdir(parents=True, exist_ok=True)
    confusion_path = paths["figures"] / "friday_confusion_matrices.png"
    comparison_path = paths["figures"] / "friday_threshold_comparison.png"
    curve_path = paths["figures"] / "friday_precision_recall.png"

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, row in zip(axes, rows, strict=True):
        grid = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        axis.imshow(grid, cmap="Blues")
        axis.set_xticks([0, 1], ["Prédit normal", "Prédit attaque"])
        axis.set_yticks([0, 1], ["Normal", "Attaque"])
        axis.set_title(f"Seuil {row['threshold']:.2f}")
        for (i, j), value in np.ndenumerate(grid):
            axis.text(j, i, f"{value:,}".replace(",", " "), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=120)
    plt.close(figure)

    plot_threshold_impact(rows, comparison_path)

    precision, recall, _ = precision_recall_curve(labels, scores)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(recall, precision, color="#1b7f4e", label=f"PR-AUC = {rows[0]['pr_auc']:.4f}")
    for row, color in zip(rows, ("#b85c38", "#0b3d2e"), strict=True):
        axis.scatter([row["recall"],], [row["precision"],], color=color, zorder=3, label=f"seuil {row['threshold']:.2f}")
    axis.set_xlabel("Rappel")
    axis.set_ylabel("Précision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title("Courbe précision-rappel sur le vendredi")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(curve_path, dpi=120)
    plt.close(figure)
    return {
        "confusion": str(confusion_path),
        "comparison": str(comparison_path),
        "precision_recall": str(curve_path),
    }


def plot_threshold_impact(rows: list[dict[str, Any]], path: Path) -> None:
    """Deux échelles distinctes, pour que les fausses alertes restent lisibles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    names = [f"{row['threshold']:.2f}" for row in rows]
    colors = ("#4c78a8", "#f58518")
    for axis, key, title in (
        (axes[0], "fp", "Fausses alertes"),
        (axes[1], "fn", "Attaques manquées"),
    ):
        axis.bar(names, [row[key] for row in rows], color=colors)
        axis.set_title(title)
        axis.set_xlabel("Seuil")
        axis.set_ylabel("Nombre de flux")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def write_friday_report(payload: dict[str, Any], path: Path) -> None:
    reference, official = payload["thresholds"]
    lines = [
        "# Évaluation finale sur le vendredi",
        "",
        "XGBoost est utilisé tel qu'il a été enregistré. Il n'a pas été réentraîné. Les seuils 0,50 et 0,01 étaient déjà fixés : 0,50 est la référence, 0,01 est le seuil officiel choisi sur le jeudi. Aucun des chiffres ci-dessous n'a servi à modifier le modèle, le seuil ou les caractéristiques. SHAP et l'audit des variables n'ont pas été lancés.",
        "",
        f"Vendredi : {_fmt_int(payload['rows'])} flux, dont {_fmt_int(payload['class_counts']['attack'])} attaques ({payload['attack_rate']:.2%}) et {_fmt_int(payload['class_counts']['normal'])} flux normaux. Familles : "
        + ", ".join(f"{name} {_fmt_int(count)}" for name, count in payload["attack_families"].items())
        + ".",
        "",
        f"PR-AUC, calculée une fois sur les probabilités continues : **{_fmt(payload['pr_auc'])}**.",
        "",
        "## Les deux seuils",
        "",
        "| | Seuil 0,50 | Seuil 0,01 |",
        "|---|---:|---:|",
    ]
    fields = [
        ("Accuracy", "accuracy"),
        ("Précision", "precision"),
        ("Rappel", "recall"),
        ("F1", "f1"),
        ("FPR", "fpr"),
        ("FNR", "fnr"),
        ("TN", "tn"),
        ("FP", "fp"),
        ("FN", "fn"),
        ("TP", "tp"),
    ]
    for label, key in fields:
        left = reference[key]
        right = official[key]
        if isinstance(left, float):
            lines.append(f"| {label} | {_fmt(left)} | {_fmt(right)} |")
        else:
            lines.append(f"| {label} | {_fmt_int(left)} | {_fmt_int(right)} |")
    delta_fp = official["fp"] - reference["fp"]
    delta_fn = official["fn"] - reference["fn"]
    precision_move = "monte" if official["precision"] > reference["precision"] else "baisse"
    lines.extend(
        [
            "",
            "## Fausses alertes et attaques manquées",
            "",
            f"Au seuil 0,50, le modèle laisse passer {_fmt_int(reference['fn'])} attaques et lève {_fmt_int(reference['fp'])} fausses alertes. Au seuil officiel 0,01, les attaques manquées passent à {_fmt_int(official['fn'])} ({delta_fn:+d}) et les fausses alertes à {_fmt_int(official['fp'])} ({delta_fp:+d}). Le rappel passe de {_fmt(reference['recall'])} à {_fmt(official['recall'])}. La précision {precision_move}, de {_fmt(reference['precision'])} à {_fmt(official['precision'])}, parce que les détections ajoutées sont en grande majorité de vraies attaques. La PR-AUC ne change pas.",
            "",
            "### Par étiquette d'origine",
            "",
            "| Étiquette | Seuil | TP | FN | FP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for threshold_row, family_rows in (
        (reference, payload["families_at_reference_threshold"]),
        (official, payload["families_at_official_threshold"]),
    ):
        for row in family_rows:
            lines.append(
                f"| {row['label']} | {threshold_row['threshold']:.2f} | {_fmt_int(row['tp'])} | {_fmt_int(row['fn'])} | {_fmt_int(row['fp'])} |"
            )
    inference = payload["inference"]
    lines.extend(
        [
            "",
            "## Temps d'inférence",
            "",
            "La mesure entoure `predict_proba` du pipeline complet, imputation comprise, avec `time.perf_counter`. Cinq passages sont faits sur tout le vendredi. La moyenne et le temps par flux utilisent ces cinq passages. Le même chronométrage est répété sur les 10 000 premiers flux du vendredi, comme prévu dans la configuration. Ces temps ne servent à aucun choix.",
            "",
            f"Vendredi complet : {', '.join(f'{item:.3f} s' for item in inference['full_seconds'])}. Moyenne {_fmt(inference['full_mean_seconds'], 3)} s, soit {inference['seconds_per_flow']:.3e} s par flux.",
            "",
            f"Lot de 10 000 flux : moyenne {_fmt(inference['batch_mean_seconds'], 3)} s, soit {inference['batch_seconds_per_flow']:.3e} s par flux.",
            "",
            "Graphiques : `outputs/figures/friday_confusion_matrices.png`, `outputs/figures/friday_threshold_comparison.png`, `outputs/figures/friday_precision_recall.png`.",
            "",
            "Résultats : `outputs/metrics/friday_evaluation.json` et `outputs/tables/friday_evaluation.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_friday_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Score le vendredi aux deux seuils figés. N'écrit pas dans le fichier du modèle."""
    if config["phases"].get("explanation") != "blocked" or config["phases"].get("audit") != "blocked":
        raise RuntimeError("SHAP et l'audit restent fermés pendant cette évaluation.")
    if config["evaluation"]["official_test_reads"] != "single_final_pass":
        raise RuntimeError("Le test ne s'évalue qu'une fois.")
    paths = resolve_paths(config)
    training = json.loads((paths["metrics"] / "training_validation.json").read_text(encoding="utf-8"))
    threshold_payload = json.loads((paths["metrics"] / "xgboost_threshold.json").read_text(encoding="utf-8"))
    if training["selected_model"] != "xgboost":
        raise RuntimeError("Le modèle évalué n'est pas le modèle retenu.")
    if abs(float(threshold_payload["selected_threshold"]) - OFFICIAL_THRESHOLD) > 1e-12:
        raise RuntimeError("Le seuil officiel enregistré n'est pas 0,01.")
    features = list(training["kept_features"])
    model_path = paths["models"] / "xgboost.joblib"
    model_bytes_before = model_path.stat().st_size
    model = load(model_path)
    if int(model.named_steps["classifier"].n_features_in_) != len(features):
        raise RuntimeError("Le modèle n'attend pas les 61 caractéristiques retenues.")
    loaded = _load_friday(features, paths["interim"] / "flows.parquet")
    repeats = int(config["timing"]["inference_repeats"])
    batch_rows = int(config["timing"]["inference_batch_rows"])
    full_times, scores = _time_predictions(model, loaded["features"], repeats)
    batch_times, _ = _time_predictions(model, loaded["features"][:batch_rows], repeats)
    if model_path.stat().st_size != model_bytes_before:
        raise RuntimeError("Le fichier du modèle a changé pendant l'évaluation.")
    pr_auc = float(average_precision_score(loaded["labels"], scores))
    rows = []
    for threshold in (REFERENCE_THRESHOLD, OFFICIAL_THRESHOLD):
        metrics = _metrics(loaded["labels"], scores, threshold)
        metrics["pr_auc"] = pr_auc
        rows.append(metrics)
    families = {
        label: int(np.sum(loaded["originals"] == label))
        for label in sorted(set(loaded["originals"].tolist()))
        if label != "BENIGN"
    }
    manifest = json.loads((paths["metrics"] / "preparation_manifest.json").read_text(encoding="utf-8"))
    if families != manifest["chronological"]["families"]["test"]:
        raise RuntimeError("Les familles du vendredi ne correspondent pas au manifeste.")
    payload = {
        "model": "xgboost",
        "model_modified": False,
        "model_refit": False,
        "threshold_adjusted_on_test": False,
        "shap_computed": False,
        "feature_audit_started": False,
        "rows": int(len(loaded["labels"])),
        "class_counts": {
            "normal": int(np.sum(loaded["labels"] == 0)),
            "attack": int(np.sum(loaded["labels"] == 1)),
        },
        "attack_rate": float(np.mean(loaded["labels"] == 1)),
        "attack_families": families,
        "pr_auc": pr_auc,
        "thresholds": rows,
        "families_at_official_threshold": _family_table(
            loaded["originals"], loaded["labels"], scores, OFFICIAL_THRESHOLD
        ),
        "families_at_reference_threshold": _family_table(
            loaded["originals"], loaded["labels"], scores, REFERENCE_THRESHOLD
        ),
        "inference": {
            "method": "time.perf_counter autour de pipeline.predict_proba, cinq passages",
            "full_seconds": [float(item) for item in full_times],
            "full_mean_seconds": float(np.mean(full_times)),
            "seconds_per_flow": float(np.mean(full_times) / len(loaded["labels"])),
            "batch_rows": batch_rows,
            "batch_seconds": [float(item) for item in batch_times],
            "batch_mean_seconds": float(np.mean(batch_times)),
            "batch_seconds_per_flow": float(np.mean(batch_times) / batch_rows),
        },
    }
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    figures = _plots(loaded["labels"], scores, rows, paths)
    payload["figures"] = figures
    result = paths["metrics"] / "friday_evaluation.json"
    temporary = result.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(result)
    _write_csv(paths["tables"] / "friday_evaluation.csv", rows, pr_auc)
    write_friday_report(payload, paths["project_root"] / "docs" / "friday_evaluation.md")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], pr_auc: float) -> None:
    header = ["threshold", "role", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "pr_auc"]
    roles = ["reference_0_50", "official_0_01"]
    lines = [",".join(header)]
    for role, row in zip(roles, rows, strict=True):
        values = [
            f"{row['threshold']:.2f}",
            role,
            str(row["tn"]),
            str(row["fp"]),
            str(row["fn"]),
            str(row["tp"]),
            repr(row["accuracy"]),
            repr(row["precision"]),
            repr(row["recall"]),
            repr(row["f1"]),
            repr(row["fpr"]),
            repr(row["fnr"]),
            repr(pr_auc),
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
