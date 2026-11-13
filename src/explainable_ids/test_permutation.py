"""Permutation importance du XGBoost officiel sur tout le vendredi.

Lecture explicative, après l'évaluation finale. Aucun ajustement, aucun
seuil, aucun fichier officiel n'est réécrit. Le vendredi ne choisit rien.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from explainable_ids.config import resolve_paths
from explainable_ids.friday_eval import EXPECTED_TEST_ROWS, _load_friday

REPEATS = 5
SEED = 42
TOP_K = 10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranks(order: list[str]) -> dict[str, int]:
    return {name: rank for rank, name in enumerate(order, start=1)}


def _read_shap(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = [
        {
            "feature": row["feature"],
            "mean_abs_shap": float(row["mean_abs_shap"]),
        }
        for row in rows
    ]
    parsed.sort(key=lambda item: (-item["mean_abs_shap"], item["feature"]))
    return parsed


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            cells.append(repr(value) if isinstance(value, float) else str(value))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_importance_report(payload: dict[str, Any], path: Path) -> None:
    comparison = payload["top10_side_by_side"]
    lines = [
        "# Comparaison des trois importances",
        "",
        "Cette lecture est postérieure à l'évaluation finale. Elle ne réentraîne rien, ne change aucun hyperparamètre et ne change pas le seuil 0,01. Elle ne sélectionne aucune variable. Les fichiers de l'évaluation du vendredi ne sont pas réécrits.",
        "",
        "## Méthode",
        "",
        f"L'impureté est celle de la forêt officielle déjà entraînée, lue dans `outputs/models/random_forest.joblib`, sur les {payload['n_features']} caractéristiques. Elle ne dépend pas du vendredi.",
        "",
        f"La permutation porte sur les {payload['rows']} flux du vendredi, soit tout le test. La métrique est la PR-AUC, `average_precision_score` sur `predict_proba` de la classe attaque. Chaque variable est mélangée {payload['n_repeats']} fois, graine {payload['seed']}, avec `n_jobs=1`. L'importance est la baisse moyenne de PR-AUC. Le modèle est le XGBoost officiel, imputation comprise, sans nouvel ajustement. Le calcul a duré {payload['seconds']:.1f} s. La PR-AUC de départ, {payload['baseline_average_precision']:.6f}, reproduit celle déjà publiée, {payload['published_average_precision']:.6f}.",
        "",
        f"Le SHAP est celui déjà calculé sur {payload['shap_rows']} flux du vendredi, graine {payload['shap_seed']}. Le classement est la moyenne des valeurs absolues, en log-odds de l'attaque. Il n'est pas recalculé.",
        "",
        "Le coût d'une prédiction sur tout le vendredi est d'environ une demi-seconde. Cinq répétitions sur 61 variables font environ 300 prédictions, soit quelques minutes. Le test entier a donc été utilisé, sans sous-échantillon.",
        "",
        "## Dix premières variables",
        "",
        "| Rang | Forêt, impureté | Permutation, vendredi | SHAP, moyenne absolue |",
        "|---:|---|---|---|",
    ]
    for index in range(TOP_K):
        lines.append(
            f"| {index + 1} | {comparison[index]['random_forest']} | {comparison[index]['permutation_test']} | {comparison[index]['shap_test']} |"
        )
    lines.extend(
        [
            "",
            "## Rangs des variables présentes dans au moins un top 10",
            "",
            "| Variable | Forêt | Permutation vendredi | SHAP vendredi |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["union_ranks"]:
        lines.append(
            f"| {row['feature']} | {row['rank_random_forest']} | {row['rank_permutation_test']} | {row['rank_shap_test']} |"
        )
    lines.extend(
        [
            "",
            "## Lecture",
            "",
            payload["reading"],
            "",
            "## Limites",
            "",
            "La forêt a été apprise du lundi au mercredi. Sa mesure d'impureté décrit cet apprentissage, pas le vendredi. La permutation et le SHAP décrivent le vendredi, mais pas les mêmes lignes : la permutation voit les 595 723 flux, le SHAP un échantillon de 2 000. Une baisse de PR-AUC dit qu'une variable sert au classement sur ce jour. Elle ne dit pas qu'il faudrait la retirer, et elle ne change pas le modèle officiel. Des variables corrélées, surtout les longueurs de paquets, se partagent l'importance : le rang de l'une peut monter quand l'autre est mélangée. Le seuil 0,01 n'entre pas dans la PR-AUC.",
            "",
            "## Fichiers",
            "",
            "- `outputs/metrics/test_permutation_importance.json`",
            "- `outputs/tables/test_permutation_importance.csv`",
            "- `outputs/tables/importance_methods_comparison.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _reading(union: list[dict[str, Any]], permutation_top: list[str], shap_top: list[str], forest_top: list[str]) -> str:
    shared = [name for name in permutation_top if name in shap_top]
    only_forest = [name for name in forest_top if name not in permutation_top and name not in shap_top]
    only_permutation = [name for name in permutation_top if name not in shap_top and name not in forest_top]
    port = next(row for row in union if row["feature"] == "Destination Port")
    return (
        f"Destination Port est au rang {port['rank_permutation_test']} de la permutation du vendredi "
        f"et au rang {port['rank_shap_test']} du SHAP, contre le rang {port['rank_random_forest']} de la forêt. "
        f"Les deux lectures du vendredi partagent {len(shared)} variables dans leur top 10 : {', '.join(shared)}. "
        f"La forêt place en tête des tailles de paquets qui ne sont pas dans les deux autres tops 10 : {', '.join(only_forest)}. "
        f"La permutation du vendredi fait entrer, hors du top 10 de la forêt et du SHAP : {', '.join(only_permutation) or 'aucune'}."
    )


def run_test_permutation(config: dict[str, Any]) -> dict[str, Any]:
    """Mesure la baisse de PR-AUC quand chaque variable du vendredi est mélangée."""
    paths = resolve_paths(config)
    watched = [
        paths["models"] / "xgboost.joblib",
        paths["models"] / "random_forest.joblib",
        paths["metrics"] / "friday_evaluation.json",
        paths["metrics"] / "xgboost_threshold.json",
        paths["metrics"] / "training_validation.json",
        paths["metrics"] / "shap_explanation.json",
        paths["tables"] / "shap_global_importance.csv",
    ]
    hashes = {path: _sha256_file(path) for path in watched}
    training = json.loads((paths["metrics"] / "training_validation.json").read_text(encoding="utf-8"))
    friday = json.loads((paths["metrics"] / "friday_evaluation.json").read_text(encoding="utf-8"))
    names = list(training["kept_features"])
    loaded = _load_friday(names, paths["interim"] / "flows.parquet")
    if len(loaded["labels"]) != EXPECTED_TEST_ROWS:
        raise RuntimeError("Le vendredi relu n'a pas l'effectif officiel.")

    from joblib import load
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score

    model = load(paths["models"] / "xgboost.joblib")
    if model.named_steps["classifier"].get_params()["device"] != "cpu":
        raise RuntimeError("Le modèle officiel n'est pas sur CPU.")

    def average_precision(estimator, features, target) -> float:
        scores = estimator.predict_proba(features)[:, 1]
        return float(average_precision_score(target, scores))

    started = time.perf_counter()
    permutation = permutation_importance(
        model,
        loaded["features"],
        loaded["labels"].astype(int),
        scoring=average_precision,
        n_repeats=REPEATS,
        random_state=SEED,
        n_jobs=1,
    )
    seconds = time.perf_counter() - started
    baseline = average_precision(model, loaded["features"], loaded["labels"].astype(int))
    published = float(friday["pr_auc"])
    if abs(baseline - published) > 1e-6:
        raise RuntimeError(
            f"La PR-AUC relue vaut {baseline:.8f}, pas la valeur publiée {published:.8f}."
        )

    forest = load(paths["models"] / "random_forest.joblib")
    impurity = np.asarray(forest.named_steps["classifier"].feature_importances_, dtype=np.float64)
    if impurity.shape != (len(names),):
        raise RuntimeError("L'impureté de la forêt ne couvre pas les 61 caractéristiques.")
    forest_order = [names[index] for index in np.argsort(-impurity, kind="mergesort")]
    shap_rows = _read_shap(paths["tables"] / "shap_global_importance.csv")
    if len(shap_rows) != len(names) or {row["feature"] for row in shap_rows} != set(names):
        raise RuntimeError("Le classement SHAP enregistré ne couvre pas les 61 caractéristiques.")
    shap_order = [row["feature"] for row in shap_rows]
    mean_drop = np.asarray(permutation.importances_mean, dtype=np.float64)
    std_drop = np.asarray(permutation.importances_std, dtype=np.float64)
    permutation_order = [names[index] for index in np.argsort(-mean_drop, kind="mergesort")]

    for path, digest in hashes.items():
        if _sha256_file(path) != digest:
            raise RuntimeError(f"Un fichier officiel a changé : {path.name}.")

    forest_rank = _ranks(forest_order)
    permutation_rank = _ranks(permutation_order)
    shap_rank = _ranks(shap_order)
    union_names = []
    for order in (forest_order, permutation_order, shap_order):
        for name in order[:TOP_K]:
            if name not in union_names:
                union_names.append(name)
    union = [
        {
            "feature": name,
            "rank_random_forest": forest_rank[name],
            "rank_permutation_test": permutation_rank[name],
            "rank_shap_test": shap_rank[name],
        }
        for name in union_names
    ]
    union.sort(key=lambda row: (row["rank_permutation_test"], row["rank_shap_test"], row["feature"]))
    detail = []
    for index, name in enumerate(names):
        detail.append(
            {
                "feature": name,
                "rank": permutation_rank[name],
                "mean_average_precision_drop": float(mean_drop[index]),
                "std_average_precision_drop": float(std_drop[index]),
                "random_forest_impurity": float(impurity[index]),
                "rank_random_forest": forest_rank[name],
                "mean_abs_shap": next(row["mean_abs_shap"] for row in shap_rows if row["feature"] == name),
                "rank_shap_test": shap_rank[name],
            }
        )
    detail.sort(key=lambda row: row["rank"])
    side_by_side = [
        {
            "rank": rank,
            "random_forest": forest_order[rank - 1],
            "permutation_test": permutation_order[rank - 1],
            "shap_test": shap_order[rank - 1],
        }
        for rank in range(1, TOP_K + 1)
    ]
    payload: dict[str, Any] = {
        "role": "analyse explicative postérieure à l'évaluation finale",
        "used_for_model_selection": False,
        "used_for_threshold": False,
        "used_for_feature_selection": False,
        "model_modified": False,
        "official_results_modified": False,
        "model": "xgboost",
        "rows": int(len(loaded["labels"])),
        "n_features": len(names),
        "scoring": "average_precision",
        "scoring_input": "predict_proba_positive_class",
        "n_repeats": REPEATS,
        "seed": SEED,
        "n_jobs": 1,
        "sample": "full_friday_test",
        "seconds": seconds,
        "baseline_average_precision": baseline,
        "published_average_precision": published,
        "shap_rows": 2000,
        "shap_seed": 42,
        "top10_side_by_side": side_by_side,
        "union_ranks": union,
        "features": detail,
    }
    payload["reading"] = _reading(union, permutation_order[:TOP_K], shap_order[:TOP_K], forest_order[:TOP_K])
    result = paths["metrics"] / "test_permutation_importance.json"
    temporary = result.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(result)
    _write_csv(
        paths["tables"] / "test_permutation_importance.csv",
        detail,
        [
            "rank",
            "feature",
            "mean_average_precision_drop",
            "std_average_precision_drop",
            "rank_random_forest",
            "rank_shap_test",
        ],
    )
    _write_csv(
        paths["tables"] / "importance_methods_comparison.csv",
        side_by_side,
        ["rank", "random_forest", "permutation_test", "shap_test"],
    )
    write_importance_report(payload, paths["project_root"] / "docs" / "importance_comparison.md")
    return payload
