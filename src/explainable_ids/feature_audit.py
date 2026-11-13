"""Audit des dix variables SHAP. Aucune variable n'est retirée.

Les trois classements officiels sont l'impureté de la forêt déjà
entraînée, la permutation sur la validation et le SHAP sur la validation.
Le vendredi est décrit, puis tenu à l'écart de la décision.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from explainable_ids.config import resolve_paths
from explainable_ids.friday_eval import EXPECTED_TEST_ROWS, FRIDAY_START
from explainable_ids.shap_run import representative_indices
from explainable_ids.train import EXPECTED_TRAIN_ROWS, EXPECTED_VALIDATION_ROWS

OFFICIAL_METHODS = ("rf_impurity", "permutation_validation", "shap_validation")
TRAIN_END = EXPECTED_TRAIN_ROWS
VALIDATION_END = EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS
TEST_END = FRIDAY_START + EXPECTED_TEST_ROWS


def official_candidates(
    rankings: dict[str, list[str]],
    top_k: int,
    minimum_methods: int,
) -> list[dict[str, Any]]:
    """Retient les variables présentes dans au moins deux tops officiels."""
    if set(rankings) != set(OFFICIAL_METHODS):
        raise ValueError("Le classement officiel n'utilise que les trois méthodes préenregistrées.")
    membership: dict[str, list[str]] = {}
    for method in OFFICIAL_METHODS:
        for name in rankings[method][:top_k]:
            membership.setdefault(name, []).append(method)
    rows = [
        {"feature": name, "n_methods": len(methods), "methods": methods}
        for name, methods in membership.items()
        if len(methods) >= minimum_methods
    ]
    rows.sort(key=lambda row: (-row["n_methods"], row["feature"]))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return str(value)


def _read_span(path: Path, columns: list[str], start: int, end: int) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    counts = [parquet.metadata.row_group(index).num_rows for index in range(parquet.metadata.num_row_groups)]
    if sum(counts) != TEST_END:
        raise RuntimeError("Le parquet préparé n'a pas le nombre de lignes validé.")
    offset = 0
    selected: list[int] = []
    for index, count in enumerate(counts):
        if offset + count > start and offset < end:
            selected.append(index)
        offset += count
    table = parquet.read_row_groups(selected, columns=columns)
    seen = sum(counts[: selected[0]])
    skip = start - seen
    length = end - start
    result: dict[str, np.ndarray] = {}
    for name in columns:
        values = table.column(name).to_numpy(zero_copy_only=False)[skip : skip + length]
        if len(values) != length:
            raise RuntimeError(f"Tranche incomplète pour {name}.")
        result[name] = values
    return result


def _stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    n = int(array.size)
    n_finite = int(finite.size)
    row: dict[str, Any] = {
        "n": n,
        "n_missing": n - n_finite,
        "missing_rate": float((n - n_finite) / n) if n else None,
    }
    if n_finite == 0:
        row.update(
            minimum=None,
            maximum=None,
            p10=None,
            median=None,
            p90=None,
            frac_neg1=None,
            frac_0=None,
            nunique=0,
            mode=None,
            mode_count=0,
            mode_share=None,
        )
        return row
    low, median, high = np.quantile(finite, [0.1, 0.5, 0.9])
    unique, counts = np.unique(finite, return_counts=True)
    mode_at = int(np.argmax(counts))
    row.update(
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        p10=float(low),
        median=float(median),
        p90=float(high),
        frac_neg1=float(np.mean(finite == -1.0)),
        frac_0=float(np.mean(finite == 0.0)),
        nunique=int(unique.size),
        mode=float(unique[mode_at]),
        mode_count=int(counts[mode_at]),
        mode_share=float(counts[mode_at] / n_finite),
    )
    return row


def _top_values(values: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return []
    unique, counts = np.unique(finite, return_counts=True)
    order = np.argsort(-counts, kind="mergesort")[:limit]
    total = float(finite.size)
    return [
        {"value": float(unique[index]), "count": int(counts[index]), "share": float(counts[index] / total)}
        for index in order
    ]


def _share(values: np.ndarray, target: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.mean(finite == target))


def _family_order(names: np.ndarray) -> list[str]:
    unique, counts = np.unique(names, return_counts=True)
    pairs = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (item[0] != "BENIGN", -item[1], str(item[0])))
    return [str(name) for name, _ in pairs]


def _split_profile(features: dict[str, np.ndarray], names: np.ndarray, columns: list[str]) -> dict[str, Any]:
    order = _family_order(names)
    masks = {family: names == family for family in order}
    attack = names != "BENIGN"
    if int(attack.sum()):
        order_with_attack = order + ["ATTACK"]
        masks["ATTACK"] = attack
    else:
        order_with_attack = order
    profile: dict[str, Any] = {"family_order": order, "features": {}}
    for column in columns:
        values = features[column]
        families = {}
        for family in order_with_attack:
            summary = _stats(values[masks[family]])
            if column == "Destination Port":
                summary["top_values"] = _top_values(values[masks[family]])
                if summary["mode"] is not None and "BENIGN" in masks and family != "BENIGN":
                    summary["same_value_share_in_benign"] = _share(values[masks["BENIGN"]], float(summary["mode"]))
            families[family] = summary
        profile["features"][column] = {"overall": _stats(values), "families": families}
    return profile


def _impossible_counts(features: dict[str, np.ndarray], columns: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in columns:
        values = np.asarray(features[column], dtype=np.float64)
        finite = np.isfinite(values)
        if column == "Destination Port":
            bad = finite & ((values < 0) | (values > 65535))
        elif "Length" in column or column == "min_seg_size_forward":
            bad = finite & (values < 0)
        elif column.endswith("/s") or "IAT" in column:
            bad = finite & (values < 0)
        elif column.startswith("Init_Win_bytes"):
            bad = finite & (values < -1)
        else:
            bad = np.zeros(len(values), dtype=bool)
        counts[column] = int(np.count_nonzero(bad))
    return counts


def _exact_duplicates(features: dict[str, np.ndarray], columns: list[str]) -> list[list[str]]:
    found = []
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            a = np.asarray(features[columns[left]], dtype=np.float64)
            b = np.asarray(features[columns[right]], dtype=np.float64)
            both_missing = ~np.isfinite(a) & ~np.isfinite(b)
            if bool(np.all((a == b) | both_missing)):
                found.append([columns[left], columns[right]])
    return found


def _correlation(features: dict[str, np.ndarray], columns: list[str], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(features[columns[0]])
    take = min(100_000, n)
    chosen = np.sort(rng.choice(n, size=take, replace=False))
    matrix = np.column_stack([np.asarray(features[column], dtype=np.float64)[chosen] for column in columns])
    for index in range(matrix.shape[1]):
        missing = ~np.isfinite(matrix[:, index])
        if not np.any(missing):
            continue
        fill = np.nanmedian(matrix[:, index])
        matrix[missing, index] = 0.0 if not np.isfinite(fill) else fill
    if np.any(np.std(matrix, axis=0) == 0):
        return []
    corr = np.corrcoef(matrix, rowvar=False)
    pairs = []
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            value = float(corr[left, right])
            if abs(value) >= 0.9:
                pairs.append({"left": columns[left], "right": columns[right], "pearson": value, "rows": int(take)})
    pairs.sort(key=lambda item: -abs(item["pearson"]))
    return pairs


def _ranks(names: list[str], ordered: list[str]) -> dict[str, int | None]:
    position = {name: rank for rank, name in enumerate(ordered, start=1)}
    return {name: position.get(name) for name in names}


def _plots(payload: dict[str, Any], figures: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    features = [row["feature"] for row in payload["audited_features"]]

    def label(name: str) -> str:
        return name.replace("\ufffd", "-")

    figure, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True)
    for axis, split in zip(axes, ("train", "validation", "test"), strict=True):
        families = payload["profiles"][split]["family_order"]
        shares = []
        captions = []
        for family in families:
            summary = payload["profiles"][split]["features"]["Destination Port"]["families"][family]
            shares.append(0.0 if summary["mode_share"] is None else summary["mode_share"])
            port = "—" if summary["mode"] is None else str(int(summary["mode"]))
            captions.append(f"{label(family)} (n={summary['n']}, port {port})")
        axis.barh(np.arange(len(families)), shares, color="#4c78a8")
        axis.set_yticks(np.arange(len(families)), captions, fontsize=7)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Part du port le plus fréquent")
        axis.set_title({"train": "Train", "validation": "Validation", "test": "Test"}[split])
        axis.invert_yaxis()
    figure.suptitle("Concentration du port de destination", fontsize=12)
    figure.tight_layout()
    destination = figures / "audit_destination_port.png"
    figure.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(figure)
    paths["destination_port"] = str(destination)

    for key, title, filename in (
        ("mode_share", "Part de la valeur la plus fréquente", "audit_mode_share.png"),
        ("frac_neg1", "Part des valeurs égales à -1", "audit_sentinel_neg1.png"),
        ("frac_0", "Part des valeurs égales à 0", "audit_zero_share.png"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16, 6))
        image = None
        for axis, split in zip(axes, ("train", "validation", "test"), strict=True):
            families = [name for name in payload["profiles"][split]["family_order"]]
            grid = np.array(
                [
                    [
                        payload["profiles"][split]["features"][feature]["families"][family][key] or 0.0
                        for family in families
                    ]
                    for feature in features
                ]
            )
            image = axis.imshow(grid, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
            axis.set_xticks(np.arange(len(families)), [label(name) for name in families], rotation=90, fontsize=7)
            axis.set_yticks(np.arange(len(features)), features, fontsize=7)
            axis.set_title({"train": "Train", "validation": "Validation", "test": "Test"}[split])
        figure.colorbar(image, ax=axes, fraction=0.02)
        figure.suptitle(title, fontsize=12)
        figure.tight_layout()
        destination = figures / filename
        figure.savefig(destination, dpi=120, bbox_inches="tight")
        plt.close(figure)
        paths[key] = str(destination)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(features))
    width = 0.25
    for offset, method, color in (
        (-width, "rf_impurity", "#4c78a8"),
        (0, "permutation_validation", "#f58518"),
        (width, "shap_validation", "#54a24b"),
    ):
        ranks = []
        for feature in features:
            rank = payload["ranks"][method][feature]
            ranks.append(62 if rank is None else rank)
        axis.barh(y + offset, ranks, height=width, color=color, label=method)
    axis.axvline(10.5, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, features)
    axis.set_xlabel("Rang (62 = hors classement)")
    axis.set_xlim(0, 66)
    axis.invert_yaxis()
    axis.legend(fontsize=8)
    axis.set_title("Rang des 10 variables SHAP du vendredi dans les trois méthodes officielles")
    figure.tight_layout()
    destination = figures / "audit_method_ranks.png"
    figure.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(figure)
    paths["ranks"] = str(destination)
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(repr(value))
            else:
                cells.append(str(value).replace(",", " ").replace("\ufffd", "-"))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Décrit les dix variables et applique la règle officielle, sans ablation."""
    if config["phases"].get("audit") != "open":
        raise RuntimeError("Phase « audit » non ouverte.")
    if config["feature_decisions"]["suspect_feature"]["test_metrics_may_decide"] is not False:
        raise RuntimeError("Une métrique de test ne peut pas désigner une variable.")
    if config["feature_decisions"]["suspect_feature"]["methods"] != list(OFFICIAL_METHODS):
        raise RuntimeError("Les trois méthodes officielles ont changé.")

    started = time.perf_counter()
    paths = resolve_paths(config)
    training = json.loads((paths["metrics"] / "training_validation.json").read_text(encoding="utf-8"))
    shap_path = paths["metrics"] / "shap_explanation.json"
    friday_path = paths["metrics"] / "friday_evaluation.json"
    threshold_path = paths["metrics"] / "xgboost_threshold.json"
    model_path = paths["models"] / "xgboost.joblib"
    forest_path = paths["models"] / "random_forest.joblib"
    hashes = {
        "xgboost": _sha256_file(model_path),
        "random_forest": _sha256_file(forest_path),
        "friday": _sha256_file(friday_path),
        "threshold": _sha256_file(threshold_path),
        "shap": _sha256_file(shap_path),
    }
    shap_payload = json.loads(shap_path.read_text(encoding="utf-8"))
    audited = [row["feature"] for row in shap_payload["top_10"]]
    if len(audited) != 10 or len(set(audited)) != 10:
        raise RuntimeError("Le classement SHAP du vendredi ne fournit pas dix variables distinctes.")
    kept = list(training["kept_features"])
    if any(name not in kept for name in audited):
        raise RuntimeError("Une variable auditée n'appartient pas aux 61 caractéristiques du modèle.")

    manifest = json.loads((paths["metrics"] / "preparation_manifest.json").read_text(encoding="utf-8"))
    parquet = paths["interim"] / "flows.parquet"
    descriptive_columns = audited + ["label", "label_original"]
    # Une seule lecture des dix variables sur les trois jours, pour les distributions.
    raw = _read_span(parquet, list(dict.fromkeys(descriptive_columns)), 0, TEST_END)
    names = np.asarray(raw["label_original"], dtype=object)
    spans = {
        "train": slice(0, TRAIN_END),
        "validation": slice(TRAIN_END, VALIDATION_END),
        "test": slice(VALIDATION_END, TEST_END),
    }
    if VALIDATION_END != FRIDAY_START:
        raise RuntimeError("La frontière du vendredi ne suit pas la validation.")
    profiles = {}
    impossible = {}
    for split, span in spans.items():
        bundle = {column: raw[column][span] for column in audited}
        original = names[span]
        profiles[split] = _split_profile(bundle, original, audited)
        impossible[split] = _impossible_counts(bundle, audited)
        observed = {
            family: int(np.sum(original == family))
            for family in profiles[split]["family_order"]
            if family != "BENIGN"
        }
        if observed != manifest["chronological"]["families"][split]:
            raise RuntimeError(f"Les familles de {split} ne correspondent pas au manifeste.")
    duplicates = _exact_duplicates({column: raw[column][spans["train"]] for column in audited}, audited)
    correlations = _correlation({column: raw[column][spans["train"]] for column in audited}, audited, 42)
    del raw

    from joblib import load
    from sklearn.metrics import average_precision_score

    forest = load(forest_path)
    impurity = np.asarray(forest.named_steps["classifier"].feature_importances_, dtype=np.float64)
    if impurity.shape != (len(kept),):
        raise RuntimeError("L'impureté de la forêt ne couvre pas les 61 caractéristiques.")
    impurity_order = [kept[index] for index in np.argsort(-impurity, kind="mergesort")]

    validation = _read_span(parquet, kept + ["label", "label_original"], TRAIN_END, VALIDATION_END)
    validation_names = np.asarray(validation["label_original"], dtype=object)
    validation_labels = np.asarray(validation["label"])
    matrix = np.column_stack([validation[column] for column in kept])
    permutation_rows = int(config["explanations"]["permutation"]["max_rows"])
    permutation_cap = int(config["explanations"]["permutation"]["hard_cap_rows"])
    if permutation_rows > permutation_cap:
        raise RuntimeError("La permutation dépasse son plafond.")
    permutation_seed = int(config["explanations"]["permutation"]["sample_seed"])
    repeats = int(config["explanations"]["permutation"]["n_repeats"])
    sample_index, sample_allocation = representative_indices(validation_names, permutation_rows, permutation_seed)
    sample_x = matrix[sample_index]
    sample_y = validation_labels[sample_index].astype(int)
    if len(np.unique(sample_y)) < 2:
        raise RuntimeError("L'échantillon de permutation ne contient qu'une classe.")

    model = load(model_path)
    if model.named_steps["classifier"].get_params()["device"] != "cpu":
        raise RuntimeError("XGBoost doit rester sur CPU.")

    def average_precision(estimator, features, target) -> float:
        scores = estimator.predict_proba(features)[:, 1]
        return float(average_precision_score(target, scores))

    from sklearn.inspection import permutation_importance

    permutation = permutation_importance(
        model,
        sample_x,
        sample_y,
        scoring=average_precision,
        n_repeats=repeats,
        random_state=permutation_seed,
        n_jobs=1,
    )
    permutation_mean = np.asarray(permutation.importances_mean, dtype=np.float64)
    permutation_std = np.asarray(permutation.importances_std, dtype=np.float64)
    permutation_order = [kept[index] for index in np.argsort(-permutation_mean, kind="mergesort")]

    shap_rows = min(int(config["explanations"]["shap_global"]["explained_rows"]), 2000)
    shap_seed = int(config["explanations"]["shap_global"]["sample_seed"])
    shap_index, shap_allocation = representative_indices(validation_names, shap_rows, shap_seed)
    import shap

    imputed = model.named_steps["imputer"].transform(matrix[shap_index])
    explainer = shap.TreeExplainer(
        model.named_steps["classifier"],
        model_output="raw",
        feature_perturbation="tree_path_dependent",
        feature_names=kept,
    )
    explained = explainer(imputed, check_additivity=False)
    values = np.asarray(explained.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != (shap_rows, len(kept)):
        raise RuntimeError(f"Forme SHAP de validation inattendue : {values.shape}.")
    base = np.asarray(explained.base_values, dtype=np.float64)
    margin = np.broadcast_to(base, (shap_rows,)) + values.sum(axis=1)
    model_margin = np.asarray(
        model.named_steps["classifier"].predict(imputed, output_margin=True),
        dtype=np.float64,
    )
    margin_error = float(np.max(np.abs(margin - model_margin)))
    if margin_error > 1e-3:
        raise RuntimeError(f"Le SHAP de validation ne reconstruit pas la marge. Écart {margin_error:.3e}.")
    mean_abs = np.mean(np.abs(values), axis=0)
    shap_order = [kept[index] for index in np.argsort(-mean_abs, kind="mergesort")]
    rankings = {
        "rf_impurity": impurity_order,
        "permutation_validation": permutation_order,
        "shap_validation": shap_order,
    }
    top_k = int(config["feature_decisions"]["suspect_feature"]["top_k"])
    minimum = int(config["feature_decisions"]["suspect_feature"]["minimum_methods_in_top_k"])
    candidates = official_candidates(rankings, top_k, minimum)

    if _sha256_file(model_path) != hashes["xgboost"] or _sha256_file(forest_path) != hashes["random_forest"]:
        raise RuntimeError("Un fichier de modèle a changé pendant l'audit.")
    if _sha256_file(friday_path) != hashes["friday"] or _sha256_file(threshold_path) != hashes["threshold"]:
        raise RuntimeError("Un résultat figé a changé pendant l'audit.")
    if _sha256_file(shap_path) != hashes["shap"]:
        raise RuntimeError("Le résultat SHAP du vendredi a été modifié.")

    rank_table = {method: _ranks(audited, order) for method, order in rankings.items()}
    payload: dict[str, Any] = {
        "model_modified": False,
        "model_refit": False,
        "variables_removed": [],
        "ablation_run": False,
        "stability_started": False,
        "thresholds_modified": False,
        "friday_results_modified": False,
        "test_metrics_used_for_decision": False,
        "audited_features": shap_payload["top_10"],
        "audit_source": "shap_test_illustration_top_10",
        "official_rankings_top_10": {method: order[:top_k] for method, order in rankings.items()},
        "ranks": rank_table,
        "official_candidates": candidates,
        "official_rule": {
            "top_k": top_k,
            "minimum_methods": minimum,
            "methods": list(OFFICIAL_METHODS),
            "domain_reason_required": True,
            "test_excluded_from_rankings": True,
        },
        "profiles": profiles,
        "impossible_value_counts": impossible,
        "exact_duplicates_on_train": duplicates,
        "high_pearson_on_train_subsample": correlations,
        "permutation": {
            "split": "validation",
            "rows": permutation_rows,
            "repeats": repeats,
            "seed": permutation_seed,
            "scoring": "average_precision",
            "n_attack": int(np.sum(sample_y == 1)),
            "n_normal": int(np.sum(sample_y == 0)),
            "allocation": sample_allocation,
            "top_10_std": [
                {"feature": name, "mean": float(permutation_mean[kept.index(name)]), "std": float(permutation_std[kept.index(name)])}
                for name in permutation_order[:top_k]
            ],
        },
        "shap_validation": {
            "rows": shap_rows,
            "seed": shap_seed,
            "attack_rows": int(np.sum(validation_labels[shap_index] == 1)),
            "allocation": shap_allocation,
            "scale": "mean_absolute_log_odds",
            "max_abs_margin_error": margin_error,
        },
        "impurity_note": "Importance d'impureté du Random Forest déjà enregistré, sans réentraînement.",
        "elapsed_seconds": time.perf_counter() - started,
        "hashes_unchanged": hashes,
    }
    figure_paths = _plots(payload, paths["figures"])
    payload["figures"] = figure_paths

    paths["metrics"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    result = paths["metrics"] / "feature_audit.json"
    temporary = result.with_suffix(".json.partial")
    temporary.write_text(json.dumps(_jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(result)
    np.save(paths["metrics"] / "shap_values_validation_sample.npy", values.astype(np.float64))

    summary_rows = []
    long_rows = []
    for feature in audited:
        summary_rows.append(
            {
                "feature": feature,
                "shap_test_rank": next(row["rank"] for row in shap_payload["top_10"] if row["feature"] == feature),
                "rf_rank": rank_table["rf_impurity"][feature] or "",
                "permutation_rank": rank_table["permutation_validation"][feature] or "",
                "shap_validation_rank": rank_table["shap_validation"][feature] or "",
                "train_nunique": profiles["train"]["features"][feature]["overall"]["nunique"],
                "train_frac_neg1": profiles["train"]["features"][feature]["overall"]["frac_neg1"],
                "train_frac_0": profiles["train"]["features"][feature]["overall"]["frac_0"],
                "train_missing_rate": profiles["train"]["features"][feature]["overall"]["missing_rate"],
                "train_impossible": impossible["train"][feature],
            }
        )
        for split in ("train", "validation", "test"):
            for family in profiles[split]["family_order"]:
                summary = profiles[split]["features"][feature]["families"][family]
                long_rows.append(
                    {
                        "split": split,
                        "family": family,
                        "feature": feature,
                        "n": summary["n"],
                        "median": summary["median"],
                        "p10": summary["p10"],
                        "p90": summary["p90"],
                        "mode": summary["mode"],
                        "mode_share": summary["mode_share"],
                        "frac_0": summary["frac_0"],
                        "frac_neg1": summary["frac_neg1"],
                        "nunique": summary["nunique"],
                        "same_value_share_in_benign": summary.get("same_value_share_in_benign", ""),
                    }
                )
    _write_csv(
        paths["tables"] / "feature_audit_summary.csv",
        summary_rows,
        [
            "feature",
            "shap_test_rank",
            "rf_rank",
            "permutation_rank",
            "shap_validation_rank",
            "train_nunique",
            "train_frac_neg1",
            "train_frac_0",
            "train_missing_rate",
            "train_impossible",
        ],
    )
    _write_csv(
        paths["tables"] / "feature_audit_by_family.csv",
        long_rows,
        [
            "split",
            "family",
            "feature",
            "n",
            "median",
            "p10",
            "p90",
            "mode",
            "mode_share",
            "frac_0",
            "frac_neg1",
            "nunique",
            "same_value_share_in_benign",
        ],
    )
    print("TOP IMPURITE", rankings["rf_impurity"][:10])
    print("TOP PERMUTATION", rankings["permutation_validation"][:10])
    print("TOP SHAP VAL", rankings["shap_validation"][:10])
    print("CANDIDATS", candidates)
    print("PERMUTATION ATTACKS", int(np.sum(sample_y == 1)), "SHAP VAL ATTACKS", int(np.sum(validation_labels[shap_index] == 1)))
    print("DUPLICATES", duplicates)
    print("CORR", correlations)
    print("IMPOSSIBLE TRAIN", impossible["train"])
    port = profiles["train"]["features"]["Destination Port"]["families"]
    for family in profiles["train"]["family_order"]:
        row = port[family]
        print(
            "PORT TRAIN",
            family,
            "n",
            row["n"],
            "mode",
            row["mode"],
            "share",
            None if row["mode_share"] is None else round(row["mode_share"], 4),
            "in_benign",
            None if row.get("same_value_share_in_benign") is None else round(row["same_value_share_in_benign"], 4),
            "nunique",
            row["nunique"],
        )
    return payload
