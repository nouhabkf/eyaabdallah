"""Entraînement chronologique des trois modèles.

La recherche d'hyperparamètres reste dans un sous-échantillon stratifié du
train, avec trois plis internes et la PR-AUC des scores continus. La
validation officielle ne participe pas à cette recherche. Le test n'est pas
lu. Chaque gagnant est réentraîné sur tout le train, puis comparé sur la
validation au seuil 0,5. Le seuil officiel n'est pas choisi ici.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import random
import shutil
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

from explainable_ids.config import resolve_paths
from explainable_ids.preprocess import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
)

EXPECTED_TRAIN_ROWS = 1_527_133
EXPECTED_VALIDATION_ROWS = 398_110
EXPECTED_KEPT_ROWS = 2_520_966


class PositiveWeightXGBClassifier(XGBClassifier):
    """XGBoost dont ``scale_pos_weight`` est calculé sur le ``y`` du ``fit``."""

    def fit(self, X, y, **kwargs):
        labels = np.asarray(y)
        positive = int(np.sum(labels == 1))
        negative = int(np.sum(labels == 0))
        if positive == 0 or negative == 0:
            raise RuntimeError("Les deux classes sont requises pour scale_pos_weight.")
        self.set_params(scale_pos_weight=float(negative / positive))
        kwargs.pop("eval_set", None)
        kwargs.pop("early_stopping_rounds", None)
        return super().fit(X, y, **kwargs)


def _log(message: str) -> None:
    print(message, flush=True)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_status() -> dict[str, int]:
    """Mémoire physique Windows, en octets."""
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("La mémoire disponible n'a pas pu être lue.")
    return {
        "total_bytes": int(status.ullTotalPhys),
        "available_bytes": int(status.ullAvailPhys),
    }


def stratified_subsample_indices(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Indices triés d'un sous-échantillon stratifié. Tout le vecteur s'il est plus court."""
    labels = np.asarray(y)
    if len(labels) <= max_rows:
        return np.arange(len(labels), dtype=np.int64)
    drawn, _ = train_test_split(
        np.arange(len(labels)),
        train_size=max_rows,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    return np.sort(np.asarray(drawn, dtype=np.int64))


def train_column_mask(matrix: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """Colonnes gardées, indices constants, indices doublons exacts du train."""
    kept: list[int] = []
    constant: list[int] = []
    duplicate: list[int] = []
    seen: dict[str, int] = {}
    for index in range(matrix.shape[1]):
        column = np.ascontiguousarray(matrix[:, index])
        finite = column[np.isfinite(column)]
        if finite.size == 0 or float(finite.min()) == float(finite.max()):
            constant.append(index)
            continue
        digest = hashlib.sha256(column.tobytes()).hexdigest()
        if digest in seen:
            duplicate.append(index)
            continue
        seen[digest] = index
        kept.append(index)
    if not kept:
        raise RuntimeError("Toutes les caractéristiques ont été écartées.")
    return np.asarray(kept, dtype=np.int64), constant, duplicate


def select_primary(rows: list[dict[str, Any]], family_order: list[str]) -> str:
    """PR-AUC de validation, puis rappel à 0,5, puis temps d'entraînement, puis l'ordre des familles."""
    order = {name: index for index, name in enumerate(family_order)}

    def sort_key(row: dict[str, Any]) -> tuple:
        return (
            -float(row["average_precision"]),
            -float(row["recall_at_0_5"]),
            float(row["train_seconds"]),
            order[row["family"]],
        )

    return sorted(rows, key=sort_key)[0]["family"]


def _continuous_average_precision(estimator, features, y_true) -> float:
    scores = estimator.predict_proba(features)[:, 1]
    return float(average_precision_score(y_true, scores))


def _prefix_params(params: dict[str, Any]) -> dict[str, Any]:
    return {f"classifier__{name}": value for name, value in params.items()}


def _strip_params(params: dict[str, Any]) -> dict[str, Any]:
    return {name.split("__", 1)[-1]: _jsonify(value) for name, value in params.items()}


def make_training_pipeline(config: dict[str, Any], family: str):
    if family == "logistic_regression":
        return build_logistic_pipeline(config)
    if family == "random_forest":
        return build_random_forest_pipeline(config)
    if family == "xgboost":
        pipeline = build_xgboost_pipeline(config)
        current = pipeline.named_steps["classifier"]
        weighted = PositiveWeightXGBClassifier()
        weighted.set_params(**current.get_params())
        weighted.set_params(verbosity=0)
        pipeline.steps[-1] = ("classifier", weighted)
        return pipeline
    raise RuntimeError(f"Famille inconnue : {family}.")


def search_hyperparameters(pipeline, family: str, config: dict[str, Any], features, labels):
    """Recherche interne au sous-échantillon. ``refit=False`` : rien n'est gardé pour la suite."""
    section = config["hyperparameters"][family]
    inner = config["hyperparameters"]["inner_cv"]
    folds = StratifiedKFold(
        n_splits=int(inner["folds"]),
        shuffle=bool(inner["shuffle"]),
        random_state=int(inner["seed"]),
    )
    common = {
        "estimator": pipeline,
        "scoring": _continuous_average_precision,
        "cv": folds,
        "refit": False,
        "n_jobs": int(inner["n_jobs"]),
        "error_score": "raise",
        "verbose": 1,
        "return_train_score": False,
    }
    if section["strategy"] == "grid":
        search = GridSearchCV(param_grid=_prefix_params(section["param_grid"]), **common)
    elif section["strategy"] == "randomized":
        search = RandomizedSearchCV(
            param_distributions=_prefix_params(section["param_distributions"]),
            n_iter=int(section["n_iter"]),
            random_state=int(config["runtime"]["numpy_seed"]),
            **common,
        )
    else:
        raise RuntimeError(f"Stratégie de recherche inconnue pour {family}.")
    started = time.perf_counter()
    search.fit(features, labels)
    elapsed = time.perf_counter() - started
    rows = []
    results = search.cv_results_
    for index in range(len(results["params"])):
        rows.append(
            {
                "params": _strip_params(results["params"][index]),
                "mean_inner_cv_average_precision": float(results["mean_test_score"][index]),
                "std_inner_cv_average_precision": float(results["std_test_score"][index]),
                "mean_fit_seconds": float(results["mean_fit_time"][index]),
            }
        )
    rows.sort(key=lambda row: -row["mean_inner_cv_average_precision"])
    return {
        "best_params": _strip_params(search.best_params_),
        "best_params_prefixed": _jsonify(search.best_params_),
        "inner_cv_average_precision": float(search.best_score_),
        "candidates": rows,
        "search_seconds": elapsed,
        "randomized_search_seed": int(config["runtime"]["numpy_seed"]),
        "refit_on_search_sample": False,
    }


def _value_counts(array) -> dict[str, int]:
    counted = pc.value_counts(array)
    values = counted.field("values").to_pylist()
    counts = counted.field("counts").to_pylist()
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


def _read_prefix(path: Path, columns: list[str], n_rows: int) -> pa.Table:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != EXPECTED_KEPT_ROWS:
        raise RuntimeError("Le parquet préparé n'a pas le nombre de lignes validé.")
    batches = []
    seen = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=100_000):
        if seen >= n_rows:
            break
        take = min(batch.num_rows, n_rows - seen)
        batches.append(batch.slice(0, take))
        seen += take
        if seen >= n_rows:
            break
    if seen != n_rows:
        raise RuntimeError(f"Préfixe lu incomplet : {seen} lignes sur {n_rows}.")
    return pa.Table.from_batches(batches)


def load_train_and_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Charge le train et la validation chronologiques. Le fichier d'indices de test n'est pas ouvert."""
    paths = resolve_paths(config)
    manifest = json.loads((paths["metrics"] / "preparation_manifest.json").read_text(encoding="utf-8"))
    train_path = paths["chronological_splits"] / "train.npy"
    validation_path = paths["chronological_splits"] / "validation.npy"
    train_idx = np.load(train_path)
    validation_idx = np.load(validation_path)
    if _sha256_file(train_path) != manifest["chronological"]["indices"]["train"]["sha256"]:
        raise RuntimeError("L'empreinte des indices de train a changé.")
    if _sha256_file(validation_path) != manifest["chronological"]["indices"]["validation"]["sha256"]:
        raise RuntimeError("L'empreinte des indices de validation a changé.")
    if len(train_idx) != EXPECTED_TRAIN_ROWS or len(validation_idx) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Les effectifs du découpage chronologique ne correspondent pas au protocole.")
    if train_idx.min() != 0 or train_idx.max() != EXPECTED_TRAIN_ROWS - 1:
        raise RuntimeError("Les indices de train ne forment pas le préfixe chronologique.")
    if validation_idx.min() != EXPECTED_TRAIN_ROWS or validation_idx.max() != EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS - 1:
        raise RuntimeError("Les indices de validation ne suivent pas le train.")
    if np.intersect1d(train_idx, validation_idx).size:
        raise RuntimeError("Un indice est à la fois dans le train et la validation.")

    features = list(manifest["feature_columns"])
    table = _read_prefix(
        paths["interim"] / "flows.parquet",
        features + ["label", "label_original"],
        EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS,
    )
    label = table.column("label").to_numpy(zero_copy_only=False)
    if set(np.unique(label).tolist()) - {0, 1}:
        raise RuntimeError("La cible binaire contient une valeur autre que 0 ou 1.")
    y_train = np.asarray(label[:EXPECTED_TRAIN_ROWS], dtype=np.int8)
    y_validation = np.asarray(label[EXPECTED_TRAIN_ROWS:], dtype=np.int8)
    _assert_counts(manifest, table, y_train, y_validation)

    matrix = np.empty((table.num_rows, len(features)), dtype=np.float32)
    for index, name in enumerate(features):
        matrix[:, index] = table.column(name).to_numpy(zero_copy_only=False)
    del table
    gc.collect()
    return {
        "features": features,
        "X_train": matrix[:EXPECTED_TRAIN_ROWS],
        "y_train": y_train,
        "X_validation": matrix[EXPECTED_TRAIN_ROWS:],
        "y_validation": y_validation,
        "train_index_sha256": manifest["chronological"]["indices"]["train"]["sha256"],
        "validation_index_sha256": manifest["chronological"]["indices"]["validation"]["sha256"],
    }


def _assert_counts(manifest: dict[str, Any], table: pa.Table, y_train: np.ndarray, y_validation: np.ndarray) -> None:
    chronological = manifest["chronological"]
    train_counts = chronological["class_counts"]["train"]
    validation_counts = chronological["class_counts"]["validation"]
    if int(np.sum(y_train == 0)) != train_counts["normal"] or int(np.sum(y_train == 1)) != train_counts["attack"]:
        raise RuntimeError("Les effectifs Normal/Attack du train ne correspondent pas au manifeste.")
    if (
        int(np.sum(y_validation == 0)) != validation_counts["normal"]
        or int(np.sum(y_validation == 1)) != validation_counts["attack"]
    ):
        raise RuntimeError("Les effectifs Normal/Attack de la validation ne correspondent pas au manifeste.")
    original = table.column("label_original")
    train_families = _families(_value_counts(original.slice(0, EXPECTED_TRAIN_ROWS)))
    validation_families = _families(
        _value_counts(original.slice(EXPECTED_TRAIN_ROWS, EXPECTED_VALIDATION_ROWS))
    )
    if train_families != chronological["families"]["train"]:
        raise RuntimeError("Les familles du train ne correspondent pas au manifeste.")
    if validation_families != chronological["families"]["validation"]:
        raise RuntimeError("Les familles de la validation ne correspondent pas au manifeste.")


def _families(counts: dict[str, int]) -> dict[str, int]:
    return {label: count for label, count in sorted(counts.items()) if label != "BENIGN"}


def validation_metrics(pipeline, features, labels, repeats: int) -> dict[str, Any]:
    times = []
    scores = None
    for _ in range(repeats):
        started = time.perf_counter()
        scores = pipeline.predict_proba(features)[:, 1]
        times.append(time.perf_counter() - started)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
    mean_seconds = float(np.mean(times))
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "confusion_at_0_5": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "inference_seconds": [float(item) for item in times],
        "inference_seconds_mean": mean_seconds,
        "seconds_per_flow": mean_seconds / len(labels),
        "curve_precision": np.asarray(curve_precision, dtype=np.float32),
        "curve_recall": np.asarray(curve_recall, dtype=np.float32),
        "scores": np.asarray(scores, dtype=np.float32),
    }


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "joblib", "matplotlib"]
    return {name: version(name) for name in names}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _plot_curves(curves: list[dict[str, Any]], prevalence: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    for curve in curves:
        axis.plot(
            curve["curve_recall"],
            curve["curve_precision"],
            label=f"{curve['family']} (AP = {curve['average_precision']:.4f})",
        )
    axis.hlines(prevalence, 0, 1, colors="0.4", linestyles="dashed", label=f"prévalence = {prevalence:.4f}")
    axis.set_xlabel("Rappel")
    axis.set_ylabel("Précision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title("Courbes précision-rappel sur la validation chronologique")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _fmt_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _format_bytes(row: dict[str, Any]) -> str:
    size = int(row["model_bytes"])
    if size < 1024**2:
        return f"{row['family']} {_fmt_int(size)} octets"
    return f"{row['family']} {size / 1024**2:.1f} Mio"


def write_training_report(payload: dict[str, Any], path: Path) -> None:
    selected = payload["selected_model"]
    rows = payload["models"]
    lines = [
        "# Entraînement et sélection sur la validation chronologique",
        "",
        "Les trois modèles ont été ajustés. Le test n'a pas été lu pour un score, un hyperparamètre ou le choix du modèle. Le seuil officiel `1 × FPR + 2 × FNR` n'a pas été choisi : les métriques binaires ci-dessous sont au seuil 0,5 demandé pour cette comparaison. SHAP et l'audit des variables n'ont pas été calculés. Le parquet préparé et les indices n'ont pas été modifiés.",
        "",
        f"Modèle retenu : **{selected}**, par PR-AUC de validation. Règle de départage, dans l'ordre : rappel au seuil 0,5 le plus haut, puis temps de réentraînement le plus court, puis l'ordre des familles dans la configuration. Cette dernière clé ne sert que si les trois critères précédents sont égaux.",
        "",
        "## Données",
        "",
        f"Train chronologique lundi–mercredi : {_fmt_int(payload['rows']['train'])} lignes, dont {_fmt_int(payload['class_counts']['train']['attack'])} attaques. Validation jeudi : {_fmt_int(payload['rows']['validation'])} lignes, dont {_fmt_int(payload['class_counts']['validation']['attack'])} attaques ({payload['validation_attack_rate']:.4%} de la validation). Les familles mesurées coïncident avec le manifeste de préparation.",
        "",
        f"Caractéristiques lues : {payload['feature_count_before']}. Conservées après filtres appris sur le train seul : {payload['feature_count_after']}. Colonnes constantes écartées : {', '.join(payload['dropped_constant']) or 'aucune'}. Colonnes doublons exactes écartées : {', '.join(payload['dropped_duplicate']) or 'aucune'}.",
        "",
        f"Sous-échantillon de recherche : {_fmt_int(payload['subsample']['rows'])} lignes du train, graine {payload['seeds']['subsample']}, dont {_fmt_int(payload['subsample']['attack'])} attaques. Trois plis stratifiés internes, graine {payload['seeds']['inner_cv']}. La validation officielle n'entre dans aucun pli.",
        "",
        "## Hyperparamètres retenus",
        "",
        "Le score de recherche est la moyenne d'average precision des trois plis, calculée avec `predict_proba` de la classe attaque. Ce score choisit les hyperparamètres. Il ne choisit pas le modèle. Le modèle est choisi plus bas, sur la validation.",
        "",
    ]
    for row in rows:
        params = ", ".join(f"{key} = {value}" for key, value in row["search"]["best_params"].items())
        lines.append(
            f"- **{row['family']}** : {params}. "
            f"PR-AUC interne = {_fmt_float(row['search']['inner_cv_average_precision'])}. "
            f"Recherche : {row['search']['search_seconds']:.1f} s. "
            f"Réentraînement sur tout le train : {row['train_seconds']:.1f} s."
        )
        if "scale_pos_weight" in row:
            lines.append(
                f"  `scale_pos_weight` calculé sur le train complet au moment du fit : {_fmt_float(row['scale_pos_weight'], 6)} "
                f"({_fmt_int(row['train_normal'])} / {_fmt_int(row['train_attack'])})."
            )
        if row.get("logistic_iterations") is not None:
            lines.append(f"  Itérations lbfgs : {row['logistic_iterations']}.")
    lines.extend(
        [
            "",
            "Ces PR-AUC internes sont calculées sur des plis du lundi–mercredi, donc sur les mêmes familles d'attaques que l'apprentissage. Elles saturent pour les trois modèles et ne servent pas à les départager. La validation du jeudi ne contient que des familles absentes du train. C'est cette PR-AUC de transfert qui choisit le modèle. L'écart entre le score interne et le score du jeudi, très marqué pour la régression logistique, décrit ce changement de familles. L'ajustement lui-même s'est terminé sans avertissement.",
            "",
            "## Comparaison sur la validation, seuil 0,5",
            "",
            "| Modèle | Retenu | PR-AUC | Précision | Rappel | F1 | TN | FP | FN | TP | Entraînement (s) | Inférence moyenne (s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        matrix = row["confusion_at_0_5"]
        mark = "oui" if row["family"] == selected else ""
        lines.append(
            "| {family} | {mark} | {ap} | {precision} | {recall} | {f1} | {tn} | {fp} | {fn} | {tp} | {train:.1f} | {infer:.3f} |".format(
                family=row["family"],
                mark=mark,
                ap=_fmt_float(row["average_precision"]),
                precision=_fmt_float(row["precision_at_0_5"]),
                recall=_fmt_float(row["recall_at_0_5"]),
                f1=_fmt_float(row["f1_at_0_5"]),
                tn=_fmt_int(matrix["tn"]),
                fp=_fmt_int(matrix["fp"]),
                fn=_fmt_int(matrix["fn"]),
                tp=_fmt_int(matrix["tp"]),
                train=row["train_seconds"],
                infer=row["inference_seconds_mean"],
            )
        )
    lines.extend(
        [
            "",
            "La colonne Retenu indique le modèle choisi par la PR-AUC. Les trois scores sont séparés, donc la règle de départage n'a pas eu à trancher. Le seuil 0,5 n'entre pas dans ce choix : il décrit un seul point de fonctionnement. Le seuil officiel, qui minimise `1 × FPR + 2 × FNR`, n'est pas encore fixé et pourra déplacer les comptes de la matrice. Il ne changera pas la PR-AUC.",
            "",
            "L'inférence est la moyenne de cinq passages `predict_proba` sur toute la validation. Le test n'a pas été chronométré. Les courbes précision-rappel sont dans `outputs/figures/validation_precision_recall.png`. La ligne de prévalence rappelle le taux d'attaque du jeudi ; ce n'est pas un quatrième modèle.",
            "",
            "## Ressources",
            "",
            f"Machine : {payload['resources']['processor']}, {payload['resources']['cpu_count']} processeurs logiques. "
            f"Mémoire physique : {payload['resources']['memory_total_bytes'] / 1024**3:.1f} Gio, "
            f"disponible au départ : {payload['resources']['memory_available_at_start_bytes'] / 1024**3:.1f} Gio, "
            f"à la fin : {payload['resources']['memory_available_at_end_bytes'] / 1024**3:.1f} Gio. "
            f"Durée totale : {payload['resources']['wall_seconds'] / 60:.1f} min. "
            f"Graines : numpy et recherche aléatoire {payload['seeds']['numpy']}, sous-échantillon {payload['seeds']['subsample']}, plis {payload['seeds']['inner_cv']}, modèles {payload['seeds']['model']}.",
            "",
            "Versions : " + ", ".join(f"{name} {value}" for name, value in payload["versions"].items()) + ".",
            "",
            "## Problèmes techniques",
            "",
        ]
    )
    if payload["warnings"]:
        lines.extend(f"- {item}" for item in payload["warnings"])
    else:
        lines.append("Aucun avertissement de convergence ni échec d'ajustement n'a été enregistré.")
    lines.extend(
        [
            "",
            "Les tailles des modèles enregistrés : " + ", ".join(_format_bytes(row) for row in rows) + ".",
            "",
            "## Ce qui reste fermé",
            "",
            "Le seuil de décision officiel, l'évaluation du vendredi, SHAP et l'audit des variables attendent une autorisation. Les trois modèles sont conservés.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _comparison_row(family: str, search: dict[str, Any], fitted, train_seconds: float, metrics: dict[str, Any], y_train: np.ndarray) -> dict[str, Any]:
    classifier = fitted.named_steps["classifier"]
    row = {
        "family": family,
        "search": search,
        "train_seconds": train_seconds,
        "average_precision": metrics["average_precision"],
        "precision_at_0_5": metrics["precision_at_0_5"],
        "recall_at_0_5": metrics["recall_at_0_5"],
        "f1_at_0_5": metrics["f1_at_0_5"],
        "confusion_at_0_5": metrics["confusion_at_0_5"],
        "inference_seconds": metrics["inference_seconds"],
        "inference_seconds_mean": metrics["inference_seconds_mean"],
        "seconds_per_flow": metrics["seconds_per_flow"],
        "train_normal": int(np.sum(y_train == 0)),
        "train_attack": int(np.sum(y_train == 1)),
    }
    if family == "xgboost":
        row["scale_pos_weight"] = float(classifier.get_params()["scale_pos_weight"])
    if family == "logistic_regression":
        iterations = getattr(classifier, "n_iter_", None)
        row["logistic_iterations"] = int(np.max(iterations)) if iterations is not None else None
    else:
        row["logistic_iterations"] = None
    return row


def run_official_training(config: dict[str, Any]) -> dict[str, Any]:
    """Entraîne les trois modèles et choisit le principal sur la validation."""
    if config["phases"]["training"] != "open":
        raise RuntimeError("Phase « training » non ouverte.")
    if config["phases"]["evaluation"] != "blocked":
        raise RuntimeError("Cette étape ne doit pas ouvrir l'évaluation finale.")
    if config["hyperparameters"]["validation_access_during_search"] is not False:
        raise RuntimeError("La recherche n'a pas le droit de lire la validation.")
    if config["hyperparameters"]["test_access"] is not False:
        raise RuntimeError("La recherche n'a pas le droit de lire le test.")
    if config["evaluation"]["official_partition"] != "chronological":
        raise RuntimeError("La partition officielle doit être chronologique.")

    seed = int(config["runtime"]["numpy_seed"])
    np.random.seed(seed)
    random.seed(seed)
    paths = resolve_paths(config)
    for directory in (paths["metrics"], paths["figures"], paths["tables"], paths["models"]):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    memory = memory_status()
    disk = shutil.disk_usage(paths["interim"].anchor or "C:/")
    _log(
        f"Mémoire disponible {memory['available_bytes'] / 1024**3:.1f} Gio, "
        f"disque libre {disk.free / 1024**3:.1f} Gio."
    )
    if memory["available_bytes"] < 3 * 1024**3:
        raise RuntimeError("Moins de 3 Gio de mémoire disponible : entraînement interrompu.")
    if disk.free < 8 * 1024**3:
        raise RuntimeError("Moins de 8 Gio de disque libre : entraînement interrompu.")

    loaded = load_train_and_validation(config)
    names = loaded["features"]
    kept, constant, duplicate = train_column_mask(loaded["X_train"])
    X_train = np.ascontiguousarray(loaded["X_train"][:, kept])
    X_validation = np.ascontiguousarray(loaded["X_validation"][:, kept])
    y_train = loaded["y_train"]
    y_validation = loaded["y_validation"]
    del loaded["X_train"], loaded["X_validation"]
    gc.collect()
    kept_names = [names[index] for index in kept]
    _log(f"Colonnes conservées : {len(kept_names)} sur {len(names)}.")

    subsample_rows = int(config["hyperparameters"]["subsample"]["max_rows"])
    hard_cap = int(config["hyperparameters"]["subsample"]["hard_cap_rows"])
    if subsample_rows > hard_cap:
        raise RuntimeError("Le sous-échantillon dépasse le plafond.")
    subsample_seed = int(config["hyperparameters"]["subsample"]["seed"])
    subsample_idx = stratified_subsample_indices(y_train, subsample_rows, subsample_seed)
    X_sub = np.ascontiguousarray(X_train[subsample_idx])
    y_sub = np.ascontiguousarray(y_train[subsample_idx])
    _log(f"Sous-échantillon : {len(y_sub)} lignes, {int(np.sum(y_sub == 1))} attaques.")

    repeats = int(config["timing"]["inference_repeats"])
    results = []
    curves = []
    warnings_seen: list[str] = []
    partial_path = paths["metrics"] / "training_validation.json"
    for family in config["models"]["families"]:
        _log(f"Recherche {family}")
        pipeline = make_training_pipeline(config, family)
        search = search_hyperparameters(pipeline, family, config, X_sub, y_sub)
        _log(f"Réentraînement {family} : {search['best_params']}")
        fresh = make_training_pipeline(config, family)
        fresh.set_params(**search["best_params_prefixed"])
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fresh.fit(X_train, y_train)
        train_seconds = time.perf_counter() - fit_started
        for item in caught:
            warnings_seen.append(f"{family} : {item.category.__name__} : {item.message}")
        _log(f"Validation {family}")
        metrics = validation_metrics(fresh, X_validation, y_validation, repeats)
        model_path = paths["models"] / f"{family}.joblib"
        dump(fresh, model_path, compress=1, protocol=4)
        score_path = paths["metrics"] / f"validation_scores_{family}.npy"
        np.save(score_path, metrics["scores"])
        row = _comparison_row(family, search, fresh, train_seconds, metrics, y_train)
        row["model_bytes"] = model_path.stat().st_size
        row["model_path"] = str(model_path)
        row["validation_scores_sha256"] = _sha256_file(score_path)
        results.append(row)
        curves.append(
            {
                "family": family,
                "average_precision": metrics["average_precision"],
                "curve_precision": metrics["curve_precision"],
                "curve_recall": metrics["curve_recall"],
            }
        )
        del fresh, metrics
        gc.collect()
        _log(f"{family} terminé en {train_seconds:.1f} s de réentraînement, PR-AUC validation {row['average_precision']:.6f}")
        _write_json(partial_path, {"models_finished": [item["family"] for item in results], "warnings": warnings_seen})

    selected = select_primary(results, list(config["models"]["families"]))
    figure_path = paths["figures"] / "validation_precision_recall.png"
    prevalence = float(np.mean(y_validation == 1))
    _plot_curves(curves, prevalence, figure_path)
    table_path = paths["tables"] / "validation_comparison.csv"
    _write_table(table_path, results, selected)

    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_partition": "chronological",
        "selected_model": selected,
        "selection_metric": "average_precision",
        "selection_split": "validation",
        "selection_input": "continuous_scores",
        "tie_break": list(config["model_selection"]["tie_break"]) + ["family_order"],
        "threshold_selected": False,
        "threshold_reported": 0.5,
        "test_scored": False,
        "test_indices_opened": False,
        "shap_computed": False,
        "feature_audit_started": False,
        "prepared_data_modified": False,
        "rows": {"train": int(len(y_train)), "validation": int(len(y_validation))},
        "class_counts": {
            "train": {"normal": int(np.sum(y_train == 0)), "attack": int(np.sum(y_train == 1))},
            "validation": {
                "normal": int(np.sum(y_validation == 0)),
                "attack": int(np.sum(y_validation == 1)),
            },
        },
        "validation_attack_rate": prevalence,
        "feature_count_before": len(names),
        "feature_count_after": len(kept_names),
        "kept_features": kept_names,
        "dropped_constant": [names[index] for index in constant],
        "dropped_duplicate": [names[index] for index in duplicate],
        "subsample": {
            "rows": int(len(y_sub)),
            "attack": int(np.sum(y_sub == 1)),
            "normal": int(np.sum(y_sub == 0)),
        },
        "seeds": {
            "numpy": seed,
            "subsample": subsample_seed,
            "inner_cv": int(config["hyperparameters"]["inner_cv"]["seed"]),
            "model": int(config["models"]["logistic_regression"]["random_state"]),
            "pythonhashseed": str(config["runtime"]["pythonhashseed"]),
        },
        "train_index_sha256": loaded["train_index_sha256"],
        "validation_index_sha256": loaded["validation_index_sha256"],
        "config_sha256": _sha256_file(paths["project_root"] / "config" / "config.yaml"),
        "versions": _package_versions(),
        "models": _public_rows(results),
        "warnings": warnings_seen,
        "figure": str(figure_path),
        "table": str(table_path),
        "resources": {
            "processor": platform.processor(),
            "cpu_count": os_cpu_count(),
            "memory_total_bytes": memory["total_bytes"],
            "memory_available_at_start_bytes": memory["available_bytes"],
            "memory_available_at_end_bytes": memory_status()["available_bytes"],
            "disk_free_at_start_bytes": disk.free,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    _write_json(partial_path, payload)
    report_path = paths["project_root"] / "docs" / "training_report.md"
    write_training_report(payload, report_path)
    _log(f"Modèle retenu : {selected}. Rapport : {report_path}")
    return payload


def os_cpu_count() -> int:
    import os

    return int(os.cpu_count() or 0)


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public = []
    for row in rows:
        copied = dict(row)
        copied["search"] = {
            key: value
            for key, value in row["search"].items()
            if key != "best_params_prefixed"
        }
        public.append(_jsonify(copied))
    return public


def _write_table(path: Path, rows: list[dict[str, Any]], selected: str) -> None:
    header = [
        "family",
        "selected",
        "average_precision",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
        "tn",
        "fp",
        "fn",
        "tp",
        "train_seconds",
        "inference_seconds_mean",
        "seconds_per_flow",
    ]
    lines = [",".join(header)]
    for row in rows:
        matrix = row["confusion_at_0_5"]
        values = [
            row["family"],
            str(row["family"] == selected).lower(),
            repr(row["average_precision"]),
            repr(row["precision_at_0_5"]),
            repr(row["recall_at_0_5"]),
            repr(row["f1_at_0_5"]),
            str(matrix["tn"]),
            str(matrix["fp"]),
            str(matrix["fn"]),
            str(matrix["tp"]),
            repr(row["train_seconds"]),
            repr(row["inference_seconds_mean"]),
            repr(row["seconds_per_flow"]),
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
