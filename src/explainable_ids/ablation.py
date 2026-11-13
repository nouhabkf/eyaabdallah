"""Réentraînement de XGBoost sans les trois variables suspectes.

Le modèle officiel n'est ni chargé ni réécrit. Le vendredi n'est pas scoré.
Le seuil n'est pas recherché à nouveau : 0,50 et 0,01 restent ceux déjà figés.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from explainable_ids.config import resolve_paths
from explainable_ids.friday_eval import _metrics
from explainable_ids.train import (
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALIDATION_ROWS,
    make_training_pipeline,
)

REMOVED_FEATURES = (
    "Destination Port",
    "Init_Win_bytes_backward",
    "Init_Win_bytes_forward",
)
VALIDATION_END = EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS
OFFICIAL_MODEL_NAME = "xgboost.joblib"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_train_validation(path: Path, columns: list[str]) -> tuple[dict[str, np.ndarray], int]:
    """Lit le train et le jeudi. La queue du dernier groupe, qui dépasse le jeudi, est jetée."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    counts = [parquet.metadata.row_group(index).num_rows for index in range(parquet.metadata.num_row_groups)]
    selected: list[int] = []
    offset = 0
    overlap = 0
    for index, count in enumerate(counts):
        if offset >= VALIDATION_END:
            break
        selected.append(index)
        if offset + count > VALIDATION_END:
            overlap = offset + count - VALIDATION_END
        offset += count
    table = parquet.read_row_groups(selected, columns=columns).slice(0, VALIDATION_END)
    if table.num_rows != VALIDATION_END:
        raise RuntimeError("La lecture train-validation n'a pas la longueur attendue.")
    result = {name: table.column(name).to_numpy(zero_copy_only=False) for name in columns}
    return result, overlap


def _published_original(training: dict[str, Any], threshold_payload: dict[str, Any]) -> dict[str, Any]:
    official = next(row for row in training["models"] if row["family"] == "xgboost")
    at_half = {
        "threshold": 0.5,
        "tn": official["confusion_at_0_5"]["tn"],
        "fp": official["confusion_at_0_5"]["fp"],
        "fn": official["confusion_at_0_5"]["fn"],
        "tp": official["confusion_at_0_5"]["tp"],
        "precision": official["precision_at_0_5"],
        "recall": official["recall_at_0_5"],
        "f1": official["f1_at_0_5"],
        "pr_auc": official["average_precision"],
    }
    at_official = dict(threshold_payload["selected"])
    return {
        "pr_auc": official["average_precision"],
        "train_seconds": official["train_seconds"],
        "scale_pos_weight": official["scale_pos_weight"],
        "hyperparameters": official["search"]["best_params"],
        "thresholds": [at_half, at_official],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["model", "threshold", "pr_auc", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["model"],
                    f"{row['threshold']:.2f}",
                    repr(row["pr_auc"]),
                    repr(row["precision"]),
                    repr(row["recall"]),
                    repr(row["f1"]),
                    str(row["tn"]),
                    str(row["fp"]),
                    str(row["fn"]),
                    str(row["tp"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_ablation_report(payload: dict[str, Any], path: Path) -> None:
    original = payload["original"]
    ablated = payload["ablated"]
    old_low = original["thresholds"][1]
    new_low = ablated["thresholds"][1]
    old_half = original["thresholds"][0]
    new_half = ablated["thresholds"][0]
    lines = [
        "# Ablation des trois variables suspectes",
        "",
        "Cette expérience réentraîne XGBoost sans Destination Port, Init_Win_bytes_backward et Init_Win_bytes_forward. Le modèle officiel, son fichier et ses résultats restent ceux déjà publiés. Le seuil n'est pas recherché à nouveau. Le vendredi n'est pas scoré. SHAP et la stabilité ne sont pas lancés.",
        "",
        "## Protocole",
        "",
        "Les lignes sont le train du lundi au mercredi et la validation du jeudi, aux effectifs déjà validés. L'imputation par médiane est ajustée dans le pipeline, sur ce train seulement. Les hyperparamètres sont ceux du réentraînement officiel : 100 arbres, profondeur 8, taux d'apprentissage 0,2, subsample 1, colsample_bytree 1, min_child_weight 1, `tree_method=hist`, CPU, graine 42. `scale_pos_weight` est recalculé sur les étiquettes de ce train, comme dans l'ajustement d'origine. L'arrêt anticipé reste désactivé.",
        "",
        f"Le temps mesuré entoure `fit` du pipeline complet, imputation comprise, avec `time.perf_counter`. C'est la même mesure que les { _fmt(original['train_seconds'], 3) } s du modèle officiel.",
        "",
        "La PR-AUC est `average_precision_score` sur `predict_proba` de la classe attaque. Précision, rappel, F1 et matrice de confusion sont calculés aux deux seuils déjà figés, 0,50 et 0,01, avec la règle score supérieur ou égal au seuil. Le dernier groupe du fichier parquet dépasse la fin du jeudi : ces lignes supplémentaires sont écartées avant la matrice, avant l'imputation et avant tout score. Aucune métrique de vendredi n'est produite.",
        "",
        "## Comparaison sur le jeudi",
        "",
        f"PR-AUC officielle : {_fmt(original['pr_auc'])}. PR-AUC sans les trois variables : {_fmt(ablated['pr_auc'])}. Écart : {_fmt(ablated['pr_auc'] - original['pr_auc'])}.",
        "",
        "| Modèle | Seuil | Précision | Rappel | F1 | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Officiel | 0,50 | {_fmt(old_half['precision'])} | {_fmt(old_half['recall'])} | {_fmt(old_half['f1'])} | "
            f"{old_half['tn']} | {old_half['fp']} | {old_half['fn']} | {old_half['tp']} |"
        ),
        (
            f"| Sans les trois variables | 0,50 | {_fmt(new_half['precision'])} | {_fmt(new_half['recall'])} | {_fmt(new_half['f1'])} | "
            f"{new_half['tn']} | {new_half['fp']} | {new_half['fn']} | {new_half['tp']} |"
        ),
        (
            f"| Officiel | 0,01 | {_fmt(old_low['precision'])} | {_fmt(old_low['recall'])} | {_fmt(old_low['f1'])} | "
            f"{old_low['tn']} | {old_low['fp']} | {old_low['fn']} | {old_low['tp']} |"
        ),
        (
            f"| Sans les trois variables | 0,01 | {_fmt(new_low['precision'])} | {_fmt(new_low['recall'])} | {_fmt(new_low['f1'])} | "
            f"{new_low['tn']} | {new_low['fp']} | {new_low['fn']} | {new_low['tp']} |"
        ),
        "",
        (
            f"Au seuil officiel 0,01, les fausses alertes passent de {old_low['fp']} à {new_low['fp']} "
            f"({new_low['fp'] - old_low['fp']:+d}) et les attaques manquées de {old_low['fn']} à {new_low['fn']} "
            f"({new_low['fn'] - old_low['fn']:+d})."
        ),
        "",
        f"Entraînement officiel : {_fmt(original['train_seconds'], 3)} s. Entraînement sans les trois variables : {_fmt(ablated['train_seconds'], 3)} s.",
        "",
        "## Lecture",
        "",
        payload["reading"],
        "",
        "Ce modèle ne remplace pas le XGBoost officiel. Le seuil 0,01 n'a pas été réoptimisé pour la nouvelle distribution de scores.",
        "",
        "## Fichiers",
        "",
        "- `outputs/metrics/ablation_suspect_features.json`",
        "- `outputs/tables/ablation_suspect_features.csv`",
        "- `outputs/models/xgboost_without_suspect_features.joblib`",
        "- `outputs/metrics/validation_scores_xgboost_without_suspects.npy`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _reading(original_auc: float, new_auc: float, old_fn: int, new_fn: int, old_fp: int, new_fp: int) -> str:
    drop = original_auc - new_auc
    if drop > 0.05:
        rank = "Le retrait enlève une part importante du classement sur le jeudi."
    elif drop > 0.01:
        rank = "Le retrait baisse le classement sur le jeudi, sans l'effondrer."
    elif drop > 0:
        rank = "Le classement sur le jeudi baisse très peu."
    else:
        rank = "Le classement sur le jeudi ne baisse pas."
    return (
        f"{rank} L'écart de PR-AUC est de {drop:.6f}. "
        f"Au seuil 0,01 déjà choisi, le nouveau modèle manque {new_fn - old_fn} attaques de plus "
        f"et produit {new_fp - old_fp} fausses alertes de plus que le modèle officiel. "
        "Le jeudi est composé d'attaques web dirigées vers un port unique et d'une fenêtre TCP très répétée : "
        "la baisse mesure ce que ces raccourcis apportaient sur cette validation. "
        "Elle ne dit pas, à elle seule, ce que le modèle ferait sur d'autres familles."
    )


def run_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """Ajuste un XGBoost neuf sur 58 caractéristiques et le compare au jeudi publié."""
    if tuple(REMOVED_FEATURES) != (
        "Destination Port",
        "Init_Win_bytes_backward",
        "Init_Win_bytes_forward",
    ):
        raise RuntimeError("La liste d'ablation n'est pas celle du rapport d'audit.")
    paths = resolve_paths(config)
    official_path = paths["models"] / OFFICIAL_MODEL_NAME
    ablation_path = paths["models"] / "xgboost_without_suspect_features.joblib"
    if ablation_path.resolve() == official_path.resolve():
        raise RuntimeError("L'ablation ne doit pas écraser le modèle officiel.")
    training_path = paths["metrics"] / "training_validation.json"
    threshold_path = paths["metrics"] / "xgboost_threshold.json"
    friday_path = paths["metrics"] / "friday_evaluation.json"
    scores_path = paths["metrics"] / "validation_scores_xgboost.npy"
    watched = {
        "model": _sha256_file(official_path),
        "training": _sha256_file(training_path),
        "threshold": _sha256_file(threshold_path),
        "friday": _sha256_file(friday_path),
        "scores": _sha256_file(scores_path),
    }
    training = json.loads(training_path.read_text(encoding="utf-8"))
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    if training["selected_model"] != "xgboost":
        raise RuntimeError("Le modèle officiel n'est pas XGBoost.")
    if abs(float(threshold_payload["selected_threshold"]) - 0.01) > 1e-12:
        raise RuntimeError("Le seuil officiel enregistré n'est pas 0,01.")
    kept = list(training["kept_features"])
    missing = [name for name in REMOVED_FEATURES if name not in kept]
    if missing:
        raise RuntimeError(f"Variable absente du modèle officiel : {missing}.")
    features = [name for name in kept if name not in REMOVED_FEATURES]
    if len(features) != len(kept) - 3:
        raise RuntimeError("Le retrait ne laisse pas 58 caractéristiques.")
    published = _published_original(training, threshold_payload)
    params = published["hyperparameters"]

    loaded, overlap = _read_train_validation(
        paths["interim"] / "flows.parquet",
        features + ["label"],
    )
    labels = np.asarray(loaded["label"])
    y_train = labels[:EXPECTED_TRAIN_ROWS].astype(np.int8)
    y_validation = labels[EXPECTED_TRAIN_ROWS:].astype(np.int8)
    if int(np.sum(y_train == 0)) != 1_324_353 or int(np.sum(y_train == 1)) != 202_780:
        raise RuntimeError("Les effectifs du train ne correspondent pas au manifeste.")
    if int(np.sum(y_validation == 0)) != 395_931 or int(np.sum(y_validation == 1)) != 2_179:
        raise RuntimeError("Les effectifs de la validation ne correspondent pas au manifeste.")
    matrix = np.column_stack([np.asarray(loaded[name], dtype=np.float32) for name in features])
    del loaded
    x_train = np.ascontiguousarray(matrix[:EXPECTED_TRAIN_ROWS])
    x_validation = np.ascontiguousarray(matrix[EXPECTED_TRAIN_ROWS:])
    del matrix
    if x_train.shape != (EXPECTED_TRAIN_ROWS, 58) or x_validation.shape != (EXPECTED_VALIDATION_ROWS, 58):
        raise RuntimeError("La matrice d'ablation n'a pas la forme attendue.")

    pipeline = make_training_pipeline(config, "xgboost")
    pipeline.set_params(**{f"classifier__{name}": value for name, value in params.items()})
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started
    weight = float(pipeline.named_steps["classifier"].get_params()["scale_pos_weight"])
    if abs(weight - float(published["scale_pos_weight"])) > 1e-12:
        raise RuntimeError("scale_pos_weight ne reproduit pas le rapport du train officiel.")
    if int(pipeline.named_steps["classifier"].n_features_in_) != 58:
        raise RuntimeError("Le modèle d'ablation n'a pas 58 caractéristiques.")
    if pipeline.named_steps["classifier"].get_params()["device"] != "cpu":
        raise RuntimeError("L'ablation doit rester sur CPU.")

    scores = np.asarray(pipeline.predict_proba(x_validation)[:, 1], dtype=np.float64)
    pr_auc = float(
        __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(
            y_validation, scores
        )
    )
    evaluated = []
    for threshold in (0.5, 0.01):
        row = _metrics(y_validation, scores, threshold)
        row["pr_auc"] = pr_auc
        evaluated.append(row)

    from joblib import dump

    dump(pipeline, ablation_path, compress=1, protocol=4)
    score_out = paths["metrics"] / "validation_scores_xgboost_without_suspects.npy"
    np.save(score_out, scores)
    for name, digest in watched.items():
        current = {
            "model": official_path,
            "training": training_path,
            "threshold": threshold_path,
            "friday": friday_path,
            "scores": scores_path,
        }[name]
        if _sha256_file(current) != digest:
            raise RuntimeError(f"Un fichier officiel a changé pendant l'ablation : {name}.")

    payload: dict[str, Any] = {
        "experiment": "remove_three_suspect_features",
        "replaces_official_model": False,
        "official_model_modified": False,
        "friday_scored": False,
        "threshold_reselected": False,
        "shap_computed": False,
        "stability_started": False,
        "removed_features": list(REMOVED_FEATURES),
        "feature_count": 58,
        "hyperparameters": params,
        "random_state": 42,
        "scale_pos_weight": weight,
        "preprocessing": "SimpleImputer median, fit on train only, inside the pipeline",
        "train_rows": EXPECTED_TRAIN_ROWS,
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "overlapping_rows_discarded_before_matrix": overlap,
        "original": {
            "pr_auc": published["pr_auc"],
            "train_seconds": published["train_seconds"],
            "scale_pos_weight": published["scale_pos_weight"],
            "hyperparameters": published["hyperparameters"],
            "thresholds": [
                {
                    "threshold": 0.5,
                    "precision": published["thresholds"][0]["precision"],
                    "recall": published["thresholds"][0]["recall"],
                    "f1": published["thresholds"][0]["f1"],
                    "tn": published["thresholds"][0]["tn"],
                    "fp": published["thresholds"][0]["fp"],
                    "fn": published["thresholds"][0]["fn"],
                    "tp": published["thresholds"][0]["tp"],
                    "pr_auc": published["pr_auc"],
                },
                {
                    "threshold": 0.01,
                    "precision": published["thresholds"][1]["precision"],
                    "recall": published["thresholds"][1]["recall"],
                    "f1": published["thresholds"][1]["f1"],
                    "tn": published["thresholds"][1]["tn"],
                    "fp": published["thresholds"][1]["fp"],
                    "fn": published["thresholds"][1]["fn"],
                    "tp": published["thresholds"][1]["tp"],
                    "pr_auc": published["pr_auc"],
                },
            ],
        },
        "ablated": {
            "pr_auc": pr_auc,
            "train_seconds": train_seconds,
            "model_path": str(ablation_path),
            "model_bytes": ablation_path.stat().st_size,
            "thresholds": evaluated,
        },
        "reading": _reading(
            published["pr_auc"],
            pr_auc,
            int(published["thresholds"][1]["fn"]),
            int(evaluated[1]["fn"]),
            int(published["thresholds"][1]["fp"]),
            int(evaluated[1]["fp"]),
        ),
    }
    result = paths["metrics"] / "ablation_suspect_features.json"
    temporary = result.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(result)
    csv_rows = []
    for model_name, block in (("official", payload["original"]), ("without_three_features", payload["ablated"])):
        for row in block["thresholds"]:
            csv_rows.append({"model": model_name, "pr_auc": block["pr_auc"], **row})
    _write_csv(paths["tables"] / "ablation_suspect_features.csv", csv_rows)
    write_ablation_report(payload, paths["project_root"] / "docs" / "ablation_suspect_features.md")
    return payload
