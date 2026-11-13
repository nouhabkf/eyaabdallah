"""SHAP du XGBoost figé, sur le vendredi, sans rien décider.

Le modèle, le seuil 0,01 et les fichiers d'évaluation ne sont pas modifiés.
Les cinq cas sont choisis sur les scores et les étiquettes, avant SHAP.
L'audit des variables n'est pas lancé.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from explainable_ids.config import resolve_paths
from explainable_ids.friday_eval import (
    FRIDAY_START,
    OFFICIAL_THRESHOLD,
    _load_friday,
    _metrics,
)

MAX_GLOBAL_ROWS = 2000
CASE_SPECS = (
    ("tp_confident", "attack_above", "highest"),
    ("tp_marginal", "attack_above", "closest_above"),
    ("fp_confident", "normal_above", "highest"),
    ("fp_marginal", "normal_above", "closest_above"),
    ("fn_marginal", "attack_below", "closest_below"),
)


def select_score_cases(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """Applique scores_and_labels_only_v1. Ne lit aucune valeur SHAP."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError("Les étiquettes et les scores n'ont pas la même longueur.")
    attack = labels == 1
    normal = labels == 0
    above = scores >= threshold
    pools = {
        "attack_above": attack & above,
        "normal_above": normal & above,
        "attack_below": attack & ~above,
    }
    selected: list[dict[str, Any]] = []
    for case_id, pool_name, mode in CASE_SPECS:
        eligible = np.flatnonzero(pools[pool_name])
        if len(eligible) == 0:
            selected.append(
                {
                    "id": case_id,
                    "present": False,
                    "index": None,
                    "mode": mode,
                    "reason": "aucun flux ne satisfait la règle",
                }
            )
            continue
        pool_scores = scores[eligible]
        if mode == "closest_above":
            target = float(np.min(pool_scores))
        else:
            target = float(np.max(pool_scores))
        tied = np.sort(eligible[pool_scores == target])
        index = int(tied[0])
        selected.append(
            {
                "id": case_id,
                "present": True,
                "index": index,
                "score": float(scores[index]),
                "label": int(labels[index]),
                "mode": mode,
                "n_eligible": int(len(eligible)),
                "n_tied": int(len(tied)),
                "tie_break": "plus petit indice du vendredi",
            }
        )
    return selected


def representative_indices(
    groups: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Tirage stratifié proportionnel, méthode du plus fort reste."""
    groups = np.asarray(groups)
    if n < 1 or n > len(groups):
        raise ValueError("La taille de l'échantillon sort du jeu disponible.")
    labels, counts = np.unique(groups, return_counts=True)
    raw = counts.astype(np.float64) / float(counts.sum()) * n
    allocation = np.floor(raw).astype(int)
    leftover = int(n - int(allocation.sum()))
    order = np.argsort(-(raw - allocation), kind="mergesort")
    for rank in range(leftover):
        allocation[int(order[rank])] += 1
    if np.any(allocation > counts):
        raise RuntimeError("L'allocation dépasse un groupe.")
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    report: list[dict[str, Any]] = []
    for label, take, total in zip(labels, allocation, counts, strict=True):
        pool = np.flatnonzero(groups == label)
        pick = rng.choice(pool, size=int(take), replace=False)
        chosen.append(np.asarray(pick, dtype=np.int64))
        report.append({"label": str(label), "rows": int(total), "sampled": int(take)})
    return np.sort(np.concatenate(chosen)), report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _positive_output(
    values: np.ndarray,
    base: np.ndarray,
    model_margin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Retient la sortie SHAP qui reconstruit la marge de la classe attaque."""
    values = np.asarray(values)
    base = np.asarray(base)
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    if values.ndim == 2:
        candidates.append(("positive_raw_margin", values, np.broadcast_to(base, (len(values),)).copy()))
    elif values.ndim == 3 and values.shape[-1] == 2:
        for class_index in (0, 1):
            class_base = base[:, class_index] if base.ndim == 2 else np.full(len(values), base[class_index])
            candidates.append((f"output_{class_index}", values[:, :, class_index], np.asarray(class_base)))
    else:
        raise RuntimeError(f"Forme SHAP inattendue : {values.shape}.")
    best: tuple[float, str, np.ndarray, np.ndarray] | None = None
    for name, class_values, class_base in candidates:
        margin = class_base + class_values.sum(axis=1)
        error = float(np.max(np.abs(margin - model_margin)))
        if best is None or error < best[0]:
            best = (error, name, class_values, class_base)
    assert best is not None
    error, name, class_values, class_base = best
    if error > 1e-3:
        raise RuntimeError(
            f"Les valeurs SHAP ne reconstruisent pas la marge du modèle. Écart maximal {error:.3e}."
        )
    return class_values, class_base, name, error


def _top_rows(
    names: list[str],
    shap_row: np.ndarray,
    feature_row: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(shap_row), kind="mergesort")
    rows = []
    for rank, position in enumerate(order[:limit], start=1):
        value = float(shap_row[position])
        if value > 0:
            direction = "vers l'attaque"
        elif value < 0:
            direction = "vers le normal"
        else:
            direction = "nulle"
        rows.append(
            {
                "rank": rank,
                "feature": names[int(position)],
                "feature_value": float(feature_row[position]),
                "shap_log_odds": value,
                "direction": direction,
            }
        )
    return rows


def _importance(names: list[str], values: np.ndarray) -> list[dict[str, Any]]:
    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    order = np.argsort(-mean_abs, kind="mergesort")
    rows = []
    for rank, position in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "feature": names[int(position)],
                "mean_abs_shap": float(mean_abs[position]),
                "mean_shap": float(mean_signed[position]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(repr(value))
            else:
                cells.append(str(value).replace(",", " "))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(
    sample_explanation: Any,
    case_explanations: list[tuple[str, Any]],
    figures: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    figures.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    shap.plots.beeswarm(sample_explanation, max_display=20, show=False)
    figure = plt.gcf()
    figure.suptitle("SHAP global — classe attaque — log-odds — 2 000 flux du vendredi", fontsize=11)
    figure.tight_layout()
    beeswarm = figures / "shap_summary_beeswarm.png"
    figure.savefig(beeswarm, dpi=120, bbox_inches="tight")
    plt.close(figure)
    paths["beeswarm"] = str(beeswarm)

    shap.plots.bar(sample_explanation, max_display=11, show=False)
    figure = plt.gcf()
    figure.suptitle("Importance moyenne absolue — top 10 — classe attaque", fontsize=11)
    figure.tight_layout()
    bars = figures / "shap_bar_top10.png"
    figure.savefig(bars, dpi=120, bbox_inches="tight")
    plt.close(figure)
    paths["bar"] = str(bars)

    for case_id, explanation in case_explanations:
        shap.plots.waterfall(explanation, max_display=10, show=False)
        figure = plt.gcf()
        figure.suptitle(f"{case_id} — contribution en log-odds de l'attaque", fontsize=11)
        figure.tight_layout()
        destination = figures / f"shap_waterfall_{case_id}.png"
        figure.savefig(destination, dpi=120, bbox_inches="tight")
        plt.close(figure)
        paths[case_id] = str(destination)
    return paths


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_shap_report(payload: dict[str, Any], path: Path) -> None:
    """Rapport lisible, construit uniquement à partir des résultats calculés."""
    method = payload["method"]
    check = payload["coherence"]
    lines = [
        "# Explications SHAP du XGBoost sur le vendredi",
        "",
        "Le modèle enregistré est relu. Il n'est pas réentraîné. Le seuil officiel reste 0,01, déjà choisi sur le jeudi. Les métriques du vendredi déjà publiées ne sont pas recalculées comme un nouveau résultat et leurs fichiers ne sont pas réécrits. Aucune variable n'est retirée. L'audit et l'analyse de stabilité ne sont pas commencés.",
        "",
        "## Méthode",
        "",
        f"L'explicateur est `shap.TreeExplainer` {method['shap_version']}, sur CPU, avec `feature_perturbation=tree_path_dependent` et `model_output=raw`. KernelSHAP n'est pas utilisé. Aucun jeu de fond n'est tiré : les effectifs déjà enregistrés dans les arbres servent de référence. La classe expliquée est l'attaque, codée 1. L'échelle est le log-odds de cette classe, la marge brute du modèle, pas la probabilité. Une valeur SHAP positive augmente le log-odds de l'attaque. La probabilité s'obtient par la fonction sigmoïde de la marge reconstruite.",
        "",
        f"Les scores utilisés pour classer les flux sont ceux de `predict_proba` de la classe attaque, au seuil 0,01. Cette relecture reproduit la matrice de confusion publiée : TN {payload['score_check']['tn']}, FP {payload['score_check']['fp']}, FN {payload['score_check']['fn']}, TP {payload['score_check']['tp']}. La PR-AUC relue est {_fmt(payload['score_check']['pr_auc'])}.",
        "",
        f"L'échantillon global compte {payload['sample']['rows']} flux, tirés avec la graine {payload['sample']['seed']} de façon proportionnelle aux étiquettes d'origine du vendredi. Le plafond appliqué est {MAX_GLOBAL_ROWS}. Ce tirage illustre le test. Il ne choisit ni une variable, ni un modèle, ni un seuil. Le SHAP de validation prévu pour l'audit n'est pas calculé.",
        "",
        "Les cinq cas suivent la règle `scores_and_labels_only_v1`, appliquée à tout le vendredi avant le calcul des graphiques locaux. En cas d'égalité de score, le plus petit indice du vendredi est gardé. Une catégorie vide resterait absente : aucun exemple de remplacement n'est créé.",
        "",
        f"Cohérence : la somme de la valeur de base et des contributions reconstruit la marge du modèle avec un écart maximal de {check['max_abs_margin_error']:.3e}. L'écart maximal entre la probabilité reconstruite et `predict_proba` est {check['max_abs_probability_error']:.3e}. Les décisions au seuil 0,01 coïncident sur les {check['rows_checked']} flux vérifiés.",
        "",
        f"Temps : construction de l'explicateur {payload['timing']['explainer_seconds']:.3f} s, explication des 2 000 flux {payload['timing']['global_seconds']:.3f} s, explication des cas {payload['timing']['cases_seconds']:.3f} s.",
        "",
        "## Importance globale",
        "",
        "Le classement est la moyenne des valeurs absolues SHAP sur les 2 000 flux. La moyenne signée est donnée à côté : elle dit le sens moyen, pas le rang.",
        "",
        "| Rang | Variable | Moyenne absolue | Moyenne signée |",
        "|---:|---|---:|---:|",
    ]
    for row in payload["importance"][:10]:
        lines.append(
            f"| {row['rank']} | {row['feature']} | {_fmt(row['mean_abs_shap'])} | {_fmt(row['mean_shap'])} |"
        )
    lines.extend(["", "Répartition de l'échantillon :", "", "| Étiquette | Vendredi | Échantillon |", "|---|---:|---:|"])
    for row in payload["sample"]["allocation"]:
        lines.append(f"| {row['label']} | {row['rows']} | {row['sampled']} |")
    lines.extend(["", "## Cas individuels", ""])
    for case in payload["cases"]:
        lines.append(f"### {case['id']}")
        lines.append("")
        if not case["present"]:
            lines.append(f"Catégorie absente : {case['reason']}. Aucun graphique n'a été produit.")
            lines.append("")
            continue
        true_name = "attaque" if case["label"] == 1 else "normal"
        predicted = "attaque" if case["score"] >= OFFICIAL_THRESHOLD else "normal"
        verb = "partage" if case["n_tied"] == 1 else "partagent"
        lines.append(
            f"Indice vendredi {case['index']}, ligne {case['absolute_row']} du parquet. "
            f"Étiquette d'origine : {case['label_original']}. Vraie classe : {true_name}. "
            f"Score {_fmt(case['score'], 8)}. Au seuil 0,01, la prédiction est {predicted}. "
            f"{case['n_tied']} flux {verb} ce score parmi {case['n_eligible']} éligibles ; le plus petit indice est retenu."
        )
        lines.append("")
        if case["missing_features"]:
            lines.append(
                "Valeurs manquantes avant imputation, remplacées par la médiane du train déjà ajustée : "
                + ", ".join(case["missing_features"])
                + "."
            )
            lines.append("")
        lines.append("| Rang | Variable | Valeur vue par le modèle | SHAP (log-odds) | Sens |")
        lines.append("|---:|---|---:|---:|---|")
        for item in case["contributions"]:
            lines.append(
                f"| {item['rank']} | {item['feature']} | {_fmt(item['feature_value'])} | {_fmt(item['shap_log_odds'])} | {item['direction']} |"
            )
        lines.append("")
        lines.append(f"Graphique : `{case['figure']}`.")
        lines.append("")
    lines.extend(
        [
            "## Limites",
            "",
            "Les contributions se partagent le long des chemins vus à l'entraînement. Des variables corrélées se répartissent le crédit : une forte valeur SHAP ne prouve pas qu'une variable agirait seule. L'échantillon de 2 000 flux décrit le vendredi, il ne le remplace pas. Les familles du vendredi, Bot, PortScan et DDoS, sont absentes de l'entraînement. Lorsque plusieurs flux ont exactement le même score extrême, le cas retenu est le premier dans l'ordre du fichier, pas un flux choisi pour son intérêt visuel.",
            "",
            "Ces explications ne modifient pas le modèle, le seuil, ni les métriques déjà publiées.",
            "",
            "## Fichiers",
            "",
            "- `outputs/metrics/shap_explanation.json`",
            "- `outputs/tables/shap_global_importance.csv`",
            "- `outputs/tables/shap_cases.csv`",
            "- `outputs/tables/shap_case_contributions.csv`",
            "- `outputs/figures/shap_summary_beeswarm.png`",
            "- `outputs/figures/shap_bar_top10.png`",
            "- `outputs/figures/shap_waterfall_tp_confident.png`",
            "- `outputs/figures/shap_waterfall_tp_marginal.png`",
            "- `outputs/figures/shap_waterfall_fp_confident.png`",
            "- `outputs/figures/shap_waterfall_fp_marginal.png`",
            "- `outputs/figures/shap_waterfall_fn_marginal.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_shap_explanation(config: dict[str, Any]) -> dict[str, Any]:
    """Calcule le SHAP global et les cinq cas, dans cet ordre de décision."""
    if config["phases"].get("explanation") != "open":
        raise RuntimeError("Phase « explanation » non ouverte.")
    if config["phases"].get("audit") != "blocked":
        raise RuntimeError("L'audit des variables n'est pas autorisé dans cette étape.")
    if config["explanations"]["shap_global"]["explainer"] != "tree":
        raise RuntimeError("Seul TreeExplainer est autorisé.")
    if config["explanations"]["shap_global"]["forbid_kernel_explainer"] is not True:
        raise RuntimeError("KernelSHAP est exclu.")
    if config["shap_cases"]["selection_uses_shap_values"] is not False:
        raise RuntimeError("Les cas doivent être choisis avant SHAP.")

    import shap
    from joblib import load

    paths = resolve_paths(config)
    training = json.loads((paths["metrics"] / "training_validation.json").read_text(encoding="utf-8"))
    threshold_payload = json.loads((paths["metrics"] / "xgboost_threshold.json").read_text(encoding="utf-8"))
    friday_path = paths["metrics"] / "friday_evaluation.json"
    published = json.loads(friday_path.read_text(encoding="utf-8"))
    if training["selected_model"] != "xgboost":
        raise RuntimeError("Le modèle expliqué n'est pas le modèle retenu.")
    if abs(float(threshold_payload["selected_threshold"]) - OFFICIAL_THRESHOLD) > 1e-12:
        raise RuntimeError("Le seuil officiel enregistré n'est pas 0,01.")
    requested = int(config["explanations"]["shap_global"]["explained_rows"])
    sample_size = min(requested, MAX_GLOBAL_ROWS)
    if sample_size < 1 or sample_size > MAX_GLOBAL_ROWS:
        raise RuntimeError("L'échantillon SHAP dépasse 2 000 observations.")
    seed = int(config["explanations"]["shap_global"]["sample_seed"])
    feature_names = list(training["kept_features"])

    model_path = paths["models"] / "xgboost.joblib"
    model_hash = _sha256_file(model_path)
    friday_hash = _sha256_file(friday_path)
    model = load(model_path)
    classifier = model.named_steps["classifier"]
    if classifier.get_params()["device"] != "cpu":
        raise RuntimeError("XGBoost doit rester sur CPU.")
    if int(classifier.n_features_in_) != len(feature_names):
        raise RuntimeError("Le modèle n'attend pas les 61 caractéristiques retenues.")

    loaded = _load_friday(feature_names, paths["interim"] / "flows.parquet")
    scores = np.asarray(model.predict_proba(loaded["features"])[:, 1], dtype=np.float64)
    official = next(row for row in published["thresholds"] if abs(row["threshold"] - OFFICIAL_THRESHOLD) < 1e-12)
    recomputed = _metrics(loaded["labels"], scores, OFFICIAL_THRESHOLD)
    for key in ("tn", "fp", "fn", "tp"):
        if int(recomputed[key]) != int(official[key]):
            raise RuntimeError("Les scores relus ne reproduisent pas la matrice publiée.")
    pr_auc = float(
        __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(
            loaded["labels"], scores
        )
    )
    if abs(pr_auc - float(published["pr_auc"])) > 1e-9:
        raise RuntimeError("La PR-AUC relue diffère du résultat publié.")

    cases = select_score_cases(loaded["labels"], scores, OFFICIAL_THRESHOLD)
    sample_index, allocation = representative_indices(loaded["originals"], sample_size, seed)

    imputed_sample = model.named_steps["imputer"].transform(loaded["features"][sample_index])
    present_cases = [case for case in cases if case["present"]]
    case_index = np.asarray([case["index"] for case in present_cases], dtype=np.int64)
    imputed_cases = model.named_steps["imputer"].transform(loaded["features"][case_index]) if len(case_index) else np.empty((0, len(feature_names)))

    started = time.perf_counter()
    explainer = shap.TreeExplainer(
        classifier,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
        feature_names=feature_names,
    )
    explainer_seconds = time.perf_counter() - started
    started = time.perf_counter()
    sample_raw = explainer(imputed_sample, check_additivity=True)
    global_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cases_raw = explainer(imputed_cases, check_additivity=True) if len(case_index) else None
    cases_seconds = time.perf_counter() - started

    sample_margin = np.asarray(classifier.predict(imputed_sample, output_margin=True), dtype=np.float64)
    sample_values, sample_base, output_name, margin_error = _positive_output(
        np.asarray(sample_raw.values),
        np.asarray(sample_raw.base_values),
        sample_margin,
    )
    sample_probability = _expit(sample_base + sample_values.sum(axis=1))
    model_probability = scores[sample_index]
    probability_error = float(np.max(np.abs(sample_probability - model_probability)))
    decisions_match = bool(
        np.array_equal(sample_probability >= OFFICIAL_THRESHOLD, model_probability >= OFFICIAL_THRESHOLD)
    )
    if not decisions_match or probability_error > 1e-4:
        raise RuntimeError("La probabilité reconstruite ne suit pas la prédiction du modèle.")

    import shap as shap_module

    sample_explanation = shap_module.Explanation(
        values=sample_values,
        base_values=sample_base,
        data=np.asarray(imputed_sample, dtype=np.float64),
        feature_names=feature_names,
    )
    importance = _importance(feature_names, sample_values)
    case_plots: list[tuple[str, Any]] = []
    detailed_cases: list[dict[str, Any]] = []
    if cases_raw is not None:
        case_margin = np.asarray(classifier.predict(imputed_cases, output_margin=True), dtype=np.float64)
        case_values, case_base, _, case_margin_error = _positive_output(
            np.asarray(cases_raw.values),
            np.asarray(cases_raw.base_values),
            case_margin,
        )
        margin_error = max(margin_error, case_margin_error)
        case_probability = _expit(case_base + case_values.sum(axis=1))
        if float(np.max(np.abs(case_probability - scores[case_index]))) > 1e-4:
            raise RuntimeError("Un cas local n'est pas cohérent avec son score.")
    else:
        case_values = np.empty((0, len(feature_names)))
        case_base = np.empty((0,))

    cursor = 0
    for case in cases:
        if not case["present"]:
            detailed_cases.append(case)
            continue
        raw_row = loaded["features"][case["index"]]
        missing = [feature_names[i] for i, value in enumerate(raw_row) if not np.isfinite(value)]
        explanation = shap_module.Explanation(
            values=case_values[cursor],
            base_values=float(case_base[cursor]),
            data=np.asarray(imputed_cases[cursor], dtype=np.float64),
            feature_names=feature_names,
        )
        contributions = _top_rows(feature_names, case_values[cursor], imputed_cases[cursor], 10)
        detailed = {
            **case,
            "absolute_row": int(FRIDAY_START + case["index"]),
            "label_original": str(loaded["originals"][case["index"]]),
            "missing_features": missing,
            "base_log_odds": float(case_base[cursor]),
            "reconstructed_log_odds": float(case_base[cursor] + case_values[cursor].sum()),
            "reconstructed_probability": float(case_probability[cursor]),
            "contributions": contributions,
        }
        detailed_cases.append(detailed)
        case_plots.append((case["id"], explanation))
        cursor += 1

    if _sha256_file(model_path) != model_hash:
        raise RuntimeError("Le fichier du modèle a changé pendant SHAP.")
    if _sha256_file(friday_path) != friday_hash:
        raise RuntimeError("Le résultat du vendredi a été modifié pendant SHAP.")

    figure_paths = _plots(sample_explanation, case_plots, paths["figures"])
    for case in detailed_cases:
        if case["present"]:
            case["figure"] = figure_paths[case["id"]]

    payload: dict[str, Any] = {
        "model": "xgboost",
        "model_modified": False,
        "model_refit": False,
        "threshold": OFFICIAL_THRESHOLD,
        "threshold_modified": False,
        "friday_results_modified": False,
        "feature_audit_started": False,
        "stability_started": False,
        "used_for_feature_decision": False,
        "model_sha256": model_hash,
        "friday_evaluation_sha256": friday_hash,
        "method": {
            "explainer": "shap.TreeExplainer",
            "shap_version": shap.__version__,
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw",
            "explained_class": "attack",
            "explained_class_value": 1,
            "scale": "log_odds",
            "output_selected": output_name,
            "kernel_explainer": False,
            "device": "cpu",
            "background_rows": 0,
            "case_rule": "scores_and_labels_only_v1",
            "cases_selected_before_local_shap": True,
        },
        "score_check": {
            "tn": int(recomputed["tn"]),
            "fp": int(recomputed["fp"]),
            "fn": int(recomputed["fn"]),
            "tp": int(recomputed["tp"]),
            "pr_auc": pr_auc,
            "matches_published_confusion": True,
        },
        "sample": {
            "split": "test",
            "role": "illustration_only",
            "rows": int(sample_size),
            "max_rows": MAX_GLOBAL_ROWS,
            "seed": seed,
            "stratification": "label_original_proportional_largest_remainder",
            "allocation": allocation,
            "attack_rate_sample": float(np.mean(loaded["labels"][sample_index] == 1)),
            "attack_rate_friday": float(np.mean(loaded["labels"] == 1)),
        },
        "coherence": {
            "max_abs_margin_error": margin_error,
            "max_abs_probability_error": probability_error,
            "decisions_match_at_official_threshold": decisions_match,
            "rows_checked": int(sample_size + len(case_index)),
        },
        "timing": {
            "explainer_seconds": explainer_seconds,
            "global_seconds": global_seconds,
            "cases_seconds": cases_seconds,
        },
        "importance": importance,
        "top_10": importance[:10],
        "cases": detailed_cases,
        "figures": figure_paths,
    }
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    result = paths["metrics"] / "shap_explanation.json"
    temporary = result.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(result)
    np.save(paths["metrics"] / "shap_sample_indices.npy", sample_index)
    np.save(paths["metrics"] / "shap_values_test_sample.npy", sample_values.astype(np.float64))
    _write_csv(
        paths["tables"] / "shap_global_importance.csv",
        importance,
        ["rank", "feature", "mean_abs_shap", "mean_shap"],
    )
    case_rows = []
    contribution_rows = []
    for case in detailed_cases:
        case_rows.append(
            {
                "id": case["id"],
                "present": case["present"],
                "friday_index": "" if case.get("index") is None else case["index"],
                "label_original": case.get("label_original", ""),
                "label": "" if case.get("label") is None else case.get("label"),
                "score": "" if case.get("score") is None else case.get("score"),
                "n_tied": "" if case.get("n_tied") is None else case.get("n_tied"),
            }
        )
        for item in case.get("contributions", []):
            contribution_rows.append({"id": case["id"], **item})
    _write_csv(
        paths["tables"] / "shap_cases.csv",
        case_rows,
        ["id", "present", "friday_index", "label_original", "label", "score", "n_tied"],
    )
    _write_csv(
        paths["tables"] / "shap_case_contributions.csv",
        contribution_rows,
        ["id", "rank", "feature", "feature_value", "shap_log_odds", "direction"],
    )
    write_shap_report(payload, paths["project_root"] / "docs" / "shap_report.md")
    if _sha256_file(friday_path) != friday_hash or _sha256_file(model_path) != model_hash:
        raise RuntimeError("Une écriture SHAP a modifié le modèle ou le résultat du vendredi.")
    return payload
