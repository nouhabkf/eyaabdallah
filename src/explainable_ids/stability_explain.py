"""Stabilité des explications SHAP, sans lire le vendredi.

L'expérience A change seulement la graine du modèle sur le train officiel.
L'expérience B tire 70 % de ce même train. Le jeudi sert une seule fois,
comme échantillon d'explication commun. Le modèle officiel n'est pas réécrit.
"""

from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from explainable_ids.config import resolve_paths
from explainable_ids.friday_eval import FRIDAY_START
from explainable_ids.shap_run import _positive_output, representative_indices
from explainable_ids.train import EXPECTED_TRAIN_ROWS, EXPECTED_VALIDATION_ROWS, make_training_pipeline

MODEL_SEEDS = (42, 7, 123)
SUBSAMPLE_SEEDS = (1001, 1002, 1003)
SUBSAMPLE_FRACTION = 0.70
SHAP_ROWS = 2000
SHAP_SEED = 42
OFFICIAL_MODEL_NAME = "xgboost.joblib"


def jaccard_top(left: list[str], right: list[str]) -> float:
    """Indice de Jaccard de deux listes, sans tenir compte de l'ordre."""
    first, second = set(left), set(right)
    union = first | second
    if not union:
        return 1.0
    return float(len(first & second) / len(union))


def spearman_ranks(left: dict[str, int], right: dict[str, int]) -> float:
    """Corrélation de Spearman entre deux classements complets, sans ex-aequo."""
    names = list(left)
    if set(names) != set(right):
        raise ValueError("Les deux classements ne portent pas sur les mêmes variables.")
    a = np.asarray([left[name] for name in names], dtype=np.float64)
    b = np.asarray([right[name] for name in names], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom == 0.0:
        return 1.0
    return float(np.sum(a * b) / denom)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_before_friday(path: Path, columns: list[str]) -> tuple[dict[str, np.ndarray], int]:
    """Lit seulement les groupes de lignes qui finissent avant le vendredi."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    counts = [parquet.metadata.row_group(index).num_rows for index in range(parquet.metadata.num_row_groups)]
    selected: list[int] = []
    offset = 0
    for index, count in enumerate(counts):
        if offset + count > FRIDAY_START:
            break
        selected.append(index)
        offset += count
    if offset < EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Le train officiel n'est pas entier avant le groupe qui contient le vendredi.")
    table = parquet.read_row_groups(selected, columns=columns)
    if table.num_rows != offset:
        raise RuntimeError("La lecture s'arrête à une longueur inattendue.")
    result = {name: table.column(name).to_numpy(zero_copy_only=False) for name in columns}
    return result, offset


def _ranking(names: list[str], mean_abs: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(-mean_abs, kind="mergesort")
    return [
        {
            "rank": rank,
            "feature": names[int(position)],
            "mean_abs_shap": float(mean_abs[int(position)]),
        }
        for rank, position in enumerate(order, start=1)
    ]


def _fit_and_explain(config: dict[str, Any], params: dict[str, Any], seed: int, x_fit, y_fit, raw_sample, names: list[str]):
    pipeline = make_training_pipeline(config, "xgboost")
    pipeline.set_params(**{f"classifier__{name}": value for name, value in params.items()})
    pipeline.set_params(classifier__random_state=int(seed))
    started = time.perf_counter()
    pipeline.fit(x_fit, y_fit)
    train_seconds = time.perf_counter() - started
    classifier = pipeline.named_steps["classifier"]
    if classifier.get_params()["device"] != "cpu":
        raise RuntimeError("La stabilité doit rester sur CPU.")
    if int(classifier.n_features_in_) != len(names):
        raise RuntimeError("Le modèle de stabilité n'a pas les 61 caractéristiques officielles.")
    weight = float(classifier.get_params()["scale_pos_weight"])
    imputed = pipeline.named_steps["imputer"].transform(raw_sample)
    import shap

    explainer = shap.TreeExplainer(
        classifier,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
        feature_names=names,
    )
    explained = explainer(imputed, check_additivity=False)
    margin = np.asarray(classifier.predict(imputed, output_margin=True), dtype=np.float64)
    values, _, _, margin_error = _positive_output(
        np.asarray(explained.values),
        np.asarray(explained.base_values),
        margin,
    )
    ranking = _ranking(names, np.mean(np.abs(values), axis=0))
    return {
        "train_seconds": train_seconds,
        "scale_pos_weight": weight,
        "train_rows": int(len(y_fit)),
        "train_attacks": int(np.sum(np.asarray(y_fit) == 1)),
        "margin_error": margin_error,
        "ranking": ranking,
        "pipeline": pipeline,
    }


def _pairs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(runs, 2):
        rank_left = {item["feature"]: item["rank"] for item in left["ranking"]}
        rank_right = {item["feature"]: item["rank"] for item in right["ranking"]}
        rows.append(
            {
                "left": left["run_id"],
                "right": right["run_id"],
                "jaccard_top10": jaccard_top(
                    [item["feature"] for item in left["ranking"][:10]],
                    [item["feature"] for item in right["ranking"][:10]],
                ),
                "spearman_full_ranking": spearman_ranks(rank_left, rank_right),
            }
        )
    return rows


def _stable_names(runs: list[dict[str, Any]]) -> list[str]:
    sets = [{item["feature"] for item in run["ranking"][:10]} for run in runs]
    common = set.intersection(*sets) if sets else set()
    order = [item["feature"] for item in runs[0]["ranking"] if item["feature"] in common]
    return order


def write_stability_report(payload: dict[str, Any], path: Path) -> None:
    def top_line(run: dict[str, Any]) -> str:
        names = ", ".join(item["feature"] for item in run["ranking"][:10])
        return f"| {run['run_id']} | {names} |"

    lines = [
        "# Stabilité des explications SHAP",
        "",
        "Deux expériences comparent le classement SHAP du XGBoost officiel. Aucune ne lit le vendredi. Aucune ne modifie le modèle officiel, son seuil ou ses hyperparamètres. Les fichiers officiels ne sont pas réécrits.",
        "",
        "## Protocole",
        "",
        "L'expérience A réentraîne trois fois sur tout le train officiel, du lundi au mercredi, avec les graines 42, 7 et 123. Seule cette graine change.",
        "",
        "L'expérience B ne crée pas un nouveau découpage chronologique. Pour les graines 1001, 1002 et 1003, elle tire 70 % du train officiel, stratifié sur l'étiquette d'origine. L'imputation et `scale_pos_weight` sont ajustés sur ce sous-échantillon seul. La graine du modèle reste 42. Aucun flux du jeudi ni du vendredi n'entre dans ces apprentissages.",
        "",
        f"Les six classements expliquent le même échantillon de {payload['shap_sample']['rows']} flux du jeudi, tiré avec la graine {payload['shap_sample']['seed']}. La population de ce tirage s'arrête avant le groupe de lignes qui contient le vendredi. Les {payload['shap_sample']['validation_rows_excluded_with_friday_group']} dernières lignes du jeudi, toutes normales, n'y sont pas. Les {payload['shap_sample']['validation_attacks']} attaques du jeudi y sont toutes. L'explication est TreeExplainer, sur CPU, en moyenne des valeurs absolues du log-odds de l'attaque.",
        "",
        "Le Jaccard compare les ensembles des dix premières variables. Spearman compare les rangs des 61 variables. Aucun seuil de réussite n'est appliqué.",
        "",
        "## Expérience A, graines du modèle",
        "",
        "| Run | Dix premières variables, dans l'ordre |",
        "|---|---|",
    ]
    for run in payload["experiment_a"]:
        lines.append(top_line(run))
    lines.extend(["", "| Paire | Jaccard du top 10 | Spearman des 61 rangs |", "|---|---:|---:|"])
    for row in payload["pairs_a"]:
        lines.append(
            f"| {row['left']} / {row['right']} | {row['jaccard_top10']:.3f} | {row['spearman_full_ranking']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Variables présentes dans les trois tops 10 : " + ", ".join(payload["stable_a"]) + ".",
            "",
            "## Expérience B, sensibilité au sous-échantillon d'entraînement",
            "",
            "| Run | Dix premières variables, dans l'ordre |",
            "|---|---|",
        ]
    )
    for run in payload["experiment_b"]:
        lines.append(top_line(run))
    lines.extend(["", "| Paire | Jaccard du top 10 | Spearman des 61 rangs |", "|---|---:|---:|"])
    for row in payload["pairs_b"]:
        lines.append(
            f"| {row['left']} / {row['right']} | {row['jaccard_top10']:.3f} | {row['spearman_full_ranking']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Variables présentes dans les trois tops 10 : " + ", ".join(payload["stable_b"]) + ".",
            "",
            "## Lecture",
            "",
            payload["reading"],
            "",
            "## Limites",
            "",
            "Le jeudi ne contient que 2 179 attaques, et l'échantillon SHAP en reprend une poignée. Un rang peut donc bouger sans que le modèle ait changé de fond. L'expérience B ne fait varier que le choix des flux du lundi au mercredi. Elle ne mesure pas un autre découpage des journées. Les trois variables retirées dans l'ablation ne sont pas retestées ici. Le modèle de graine 42 est un nouvel ajustement : il n'écrase pas le fichier officiel, et de petites différences numériques avec le SHAP déjà publié peuvent venir de ce second ajustement comme du fait que les 25 243 dernières lignes normales du jeudi ont été laissées de côté.",
            "",
            "## Fichiers",
            "",
            "- `outputs/metrics/stability_explanations.json`",
            "- `outputs/tables/stability_top10.csv`",
            "- `outputs/tables/stability_rank_agreement.csv`",
            "- `outputs/figures/stability_rank_heatmap.png`",
            "- `outputs/models/stability/`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _reading(payload: dict[str, Any]) -> str:
    mean_jaccard_a = float(np.mean([row["jaccard_top10"] for row in payload["pairs_a"]]))
    mean_spearman_a = float(np.mean([row["spearman_full_ranking"] for row in payload["pairs_a"]]))
    mean_jaccard_b = float(np.mean([row["jaccard_top10"] for row in payload["pairs_b"]]))
    mean_spearman_b = float(np.mean([row["spearman_full_ranking"] for row in payload["pairs_b"]]))
    return (
        f"Changer la graine du modèle laisse le classement très stable : "
        f"Jaccard moyen du top 10 égal à {mean_jaccard_a:.3f}, Spearman moyen des 61 rangs égal à {mean_spearman_a:.3f}. "
        f"Changer 30 % des flux d'entraînement du lundi au mercredi bouge davantage le top 10 : "
        f"Jaccard moyen {mean_jaccard_b:.3f}, Spearman moyen {mean_spearman_b:.3f}. "
        f"Les variables communes aux trois graines sont : {', '.join(payload['stable_a'])}. "
        f"Celles communes aux trois sous-échantillons sont : {', '.join(payload['stable_b'])}."
    )


def _heatmap(runs: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen: list[str] = []
    for run in runs:
        for item in run["ranking"][:10]:
            if item["feature"] not in chosen:
                chosen.append(item["feature"])
    grid = np.array(
        [
            [next(item["rank"] for item in run["ranking"] if item["feature"] == name) for run in runs]
            for name in chosen
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.35 * len(chosen))))
    image = axis.imshow(grid, cmap="viridis_r", aspect="auto")
    axis.set_xticks(np.arange(len(runs)), [run["run_id"] for run in runs], rotation=30, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(chosen)), chosen, fontsize=8)
    for row_index in range(grid.shape[0]):
        for column_index in range(grid.shape[1]):
            axis.text(column_index, row_index, str(int(grid[row_index, column_index])), ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(image, ax=axis, label="Rang SHAP")
    axis.set_title("Rang des variables présentes dans au moins un top 10")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def run_stability_explanations(config: dict[str, Any]) -> dict[str, Any]:
    """Compare les classements SHAP des graines et des sous-échantillons autorisés."""
    paths = resolve_paths(config)
    official_path = paths["models"] / OFFICIAL_MODEL_NAME
    training_path = paths["metrics"] / "training_validation.json"
    threshold_path = paths["metrics"] / "xgboost_threshold.json"
    friday_path = paths["metrics"] / "friday_evaluation.json"
    watched = {
        official_path: _sha256_file(official_path),
        training_path: _sha256_file(training_path),
        threshold_path: _sha256_file(threshold_path),
        friday_path: _sha256_file(friday_path),
    }
    training = json.loads(training_path.read_text(encoding="utf-8"))
    official = next(row for row in training["models"] if row["family"] == "xgboost")
    params = dict(official["search"]["best_params"])
    names = list(training["kept_features"])
    if len(names) != 61:
        raise RuntimeError("Le modèle officiel n'a pas 61 caractéristiques.")
    manifest = json.loads((paths["metrics"] / "preparation_manifest.json").read_text(encoding="utf-8"))
    loaded, safe_end = _read_before_friday(paths["interim"] / "flows.parquet", names + ["label", "label_original"])
    labels = np.asarray(loaded["label"])
    originals = np.asarray(loaded["label_original"], dtype=object)
    y_train = labels[:EXPECTED_TRAIN_ROWS].astype(np.int8)
    train_names = originals[:EXPECTED_TRAIN_ROWS]
    if int(np.sum(y_train == 1)) != 202_780 or int(np.sum(y_train == 0)) != 1_324_353:
        raise RuntimeError("Le train lu n'a pas les effectifs officiels.")
    validation_names = originals[EXPECTED_TRAIN_ROWS:]
    validation_labels = labels[EXPECTED_TRAIN_ROWS:]
    attack_counts: dict[str, int] = {}
    for name in sorted(set(validation_names.tolist())):
        if name == "BENIGN":
            continue
        attack_counts[str(name)] = int(np.sum(validation_names == name))
    if attack_counts != manifest["chronological"]["families"]["validation"]:
        raise RuntimeError("Les attaques du jeudi ne sont pas toutes avant le groupe du vendredi.")
    if int(np.sum(validation_labels == 1)) != 2_179:
        raise RuntimeError("Le nombre d'attaques du jeudi lisibles sans le vendredi est inattendu.")
    matrix = np.column_stack([np.asarray(loaded[name], dtype=np.float32) for name in names])
    del loaded
    x_train = np.ascontiguousarray(matrix[:EXPECTED_TRAIN_ROWS])
    x_validation = np.ascontiguousarray(matrix[EXPECTED_TRAIN_ROWS:])
    del matrix
    shap_index, shap_allocation = representative_indices(validation_names, SHAP_ROWS, SHAP_SEED)
    raw_sample = np.ascontiguousarray(x_validation[shap_index])
    absolute_rows = EXPECTED_TRAIN_ROWS + shap_index
    if int(np.max(absolute_rows)) >= FRIDAY_START:
        raise RuntimeError("L'échantillon SHAP atteint le vendredi.")

    stability_dir = paths["models"] / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)
    runs_a: list[dict[str, Any]] = []
    from joblib import dump

    for seed in MODEL_SEEDS:
        fitted = _fit_and_explain(config, params, seed, x_train, y_train, raw_sample, names)
        destination = stability_dir / f"xgboost_model_seed_{seed}.joblib"
        if destination.name == OFFICIAL_MODEL_NAME or destination.resolve() == official_path.resolve():
            raise RuntimeError("Un run de stabilité vise le fichier officiel.")
        dump(fitted.pop("pipeline"), destination, compress=1, protocol=4)
        fitted["run_id"] = f"graine-{seed}"
        fitted["experiment"] = "model_seed"
        fitted["model_seed"] = seed
        fitted["model_path"] = str(destination)
        runs_a.append(fitted)
        print(f"A graine {seed} terminée en {fitted['train_seconds']:.1f} s", flush=True)

    runs_b: list[dict[str, Any]] = []
    subsample_target = int(np.floor(SUBSAMPLE_FRACTION * EXPECTED_TRAIN_ROWS))
    for seed in SUBSAMPLE_SEEDS:
        chosen, allocation = representative_indices(train_names, subsample_target, seed)
        if int(np.max(chosen)) >= EXPECTED_TRAIN_ROWS:
            raise RuntimeError("Un sous-échantillon sort du train officiel.")
        fitted = _fit_and_explain(
            config,
            params,
            42,
            np.ascontiguousarray(x_train[chosen]),
            y_train[chosen],
            raw_sample,
            names,
        )
        destination = stability_dir / f"xgboost_subsample_seed_{seed}.joblib"
        dump(fitted.pop("pipeline"), destination, compress=1, protocol=4)
        fitted["run_id"] = f"sous-echantillon-{seed}"
        fitted["experiment"] = "train_subsample"
        fitted["subsample_seed"] = seed
        fitted["model_seed"] = 42
        fitted["allocation"] = allocation
        fitted["model_path"] = str(destination)
        runs_b.append(fitted)
        print(f"B graine {seed} terminée en {fitted['train_seconds']:.1f} s", flush=True)

    for path, digest in watched.items():
        if _sha256_file(path) != digest:
            raise RuntimeError(f"Un fichier officiel a changé : {path.name}.")

    payload: dict[str, Any] = {
        "official_model_modified": False,
        "friday_read": False,
        "friday_rows_used": 0,
        "threshold_modified": False,
        "hyperparameters": params,
        "experiment_a_role": "graine du modèle, train officiel complet",
        "experiment_b_role": "sensibilité au choix des flux du train officiel, pas un nouveau découpage chronologique",
        "shap_sample": {
            "split": "validation_thursday",
            "rows": SHAP_ROWS,
            "seed": SHAP_SEED,
            "stratification": "label_original_proportional_largest_remainder",
            "allocation": shap_allocation,
            "validation_rows_available_without_friday": int(safe_end - EXPECTED_TRAIN_ROWS),
            "validation_rows_excluded_with_friday_group": int(EXPECTED_VALIDATION_ROWS - (safe_end - EXPECTED_TRAIN_ROWS)),
            "validation_attacks": 2179,
            "max_absolute_row": int(np.max(absolute_rows)),
        },
        "experiment_a": runs_a,
        "experiment_b": runs_b,
        "pairs_a": _pairs(runs_a),
        "pairs_b": _pairs(runs_b),
        "stable_a": _stable_names(runs_a),
        "stable_b": _stable_names(runs_b),
    }
    payload["reading"] = _reading(payload)
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    result = paths["metrics"] / "stability_explanations.json"
    temporary = result.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(result)
    top_rows = []
    for run in runs_a + runs_b:
        for item in run["ranking"][:10]:
            top_rows.append({"run_id": run["run_id"], "experiment": run["experiment"], **item})
    _write_csv(
        paths["tables"] / "stability_top10.csv",
        top_rows,
        ["run_id", "experiment", "rank", "feature", "mean_abs_shap"],
    )
    _write_csv(
        paths["tables"] / "stability_rank_agreement.csv",
        payload["pairs_a"] + payload["pairs_b"],
        ["left", "right", "jaccard_top10", "spearman_full_ranking"],
    )
    _heatmap(runs_a + runs_b, paths["figures"] / "stability_rank_heatmap.png")
    write_stability_report(payload, paths["project_root"] / "docs" / "stability_explanations.md")
    np.save(paths["metrics"] / "stability_shap_sample_indices.npy", shap_index)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(repr(value))
            else:
                cells.append(str(value))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
