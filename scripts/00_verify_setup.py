"""Vérifie la configuration, les dossiers, les dépendances et le mode CPU.

Ce script ne télécharge aucune donnée, ne lit aucun CSV et n'appelle aucun ajustement de modèle.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explainable_ids.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_yaml,
    resolve_paths,
    validate_config,
)

DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "shap",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "PyYAML",
    "pytest",
    "joblib",
    "jupyter",
    "ipykernel",
)
IMPORT_MODULES = (
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "shap",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "yaml",
    "joblib",
    "pytest",
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = datetime.now(timezone.utc).isoformat()
    problems: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    raw_config = load_yaml(DEFAULT_CONFIG_PATH)
    config_errors = validate_config(raw_config)
    problems.extend(config_errors)

    directories: dict[str, str] = {}
    raw_csv_count = None
    raw_csv_names: list[str] = []
    if not config_errors:
        paths = resolve_paths(raw_config)
        for name, path in paths.items():
            if name == "project_root":
                continue
            if name == "venv":
                directories[name] = str(path)
                if not path.exists():
                    problems.append(f"Environnement virtuel absent : {path}")
                continue
            path.mkdir(parents=True, exist_ok=True)
            directories[name] = str(path)
        raw_dir = paths["raw"]
        raw_csv_names = sorted(item.name for item in raw_dir.glob("*.csv"))
        raw_csv_count = len(raw_csv_names)
        if raw_csv_count and raw_config["dataset"]["local_status"] == "not_downloaded":
            warnings.append(
                "Des CSV sont présents alors que la configuration indique que le dataset n'est pas téléchargé. Ils n'ont pas été lus."
            )

    if os.environ.get("PYTHONHASHSEED") != "42":
        problems.append("PYTHONHASHSEED doit valoir 42 avant le démarrage de l'interpréteur.")
    if sys.version_info[:2] != (3, 12):
        problems.append(
            f"Interpréteur {sys.version_info[0]}.{sys.version_info[1]} détecté. Le projet fixe Python 3.12."
        )

    packages = _package_versions(problems)
    for module_name in IMPORT_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - le rapport doit nommer le module manquant
            problems.append(f"Import impossible : {module_name} ({exc.__class__.__name__}).")

    cpu_report: dict[str, object] = {
        "fit_called": False,
        "csv_read": False,
        "tree_explainer_executed": False,
    }
    if packages.get("xgboost") and not str(packages["xgboost"]).startswith("INDISPONIBLE"):
        if not config_errors:
            cpu_report.update(_check_cpu_models(raw_config, problems, warnings))
    if cpu_report.get("shap_tree_explainer_present") is True:
        notes.append(
            "TreeExplainer est importable. Il n'est pas exécuté : aucun modèle n'est ajusté dans cette phase."
        )

    report = {
        "checked_at_utc": started,
        "fit_called": False,
        "dataset_read": False,
        "dataset_download_attempted": False,
        "python": sys.version,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "config_sha256": hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest(),
        "config_errors": config_errors,
        "packages": packages,
        "cpu": cpu_report,
        "directories": directories,
        "raw_csv_count": raw_csv_count,
        "raw_csv_names": raw_csv_names,
        "warnings": warnings,
        "notes": notes,
        "problems": problems,
        "status": "passed" if not problems else "failed",
    }
    if not config_errors:
        destination = resolve_paths(raw_config)["metrics"] / "setup_verification.json"
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(destination)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not problems else 1


def _package_versions(problems: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in DISTRIBUTIONS:
        try:
            versions[distribution] = distribution_version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "INDISPONIBLE"
            problems.append(f"Dépendance indisponible : {distribution}.")
    return versions


def _check_cpu_models(
    config: dict,
    problems: list[str],
    warnings: list[str],
) -> dict[str, object]:
    import matplotlib
    import shap
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    report: dict[str, object] = {"fit_called": False, "tree_explainer_executed": False}
    build_info = {str(key): _jsonable(value) for key, value in xgb.build_info().items()}
    report["xgboost_build_info"] = build_info
    if "USE_CUDA" in build_info:
        use_cuda = _as_bool(build_info["USE_CUDA"])
        report["use_cuda_key_present"] = True
    else:
        use_cuda = False
        report["use_cuda_key_present"] = False
    report["use_cuda"] = use_cuda
    nvidia_smi = shutil.which("nvidia-smi")
    report["nvidia_smi_present"] = nvidia_smi is not None
    if use_cuda and nvidia_smi is None:
        warnings.append(
            "La roue XGBoost contient du code CUDA. "
            "nvidia-smi est absent et device est forcé à cpu."
        )
    elif use_cuda and nvidia_smi is not None:
        warnings.append(
            "La roue XGBoost contient du code CUDA et nvidia-smi est présent. "
            "device reste forcé à cpu."
        )

    score_params = inspect.signature(average_precision_score).parameters
    report["average_precision_accepts_y_score"] = "y_score" in score_params
    if "y_score" not in score_params:
        problems.append("average_precision_score ne publie pas l'argument de score continu y_score.")

    logistic_cfg = config["models"]["logistic_regression"]
    forest_cfg = config["models"]["random_forest"]
    xgb_cfg = config["models"]["xgboost"]
    try:
        LogisticRegression(
            class_weight=logistic_cfg["class_weight"],
            solver=logistic_cfg["solver"],
            max_iter=logistic_cfg["max_iter"],
            random_state=logistic_cfg["random_state"],
        )
        RandomForestClassifier(
            class_weight=forest_cfg["class_weight"],
            random_state=forest_cfg["random_state"],
            n_jobs=config["compute"]["sklearn_n_jobs"],
        )
        xgb_model = xgb.XGBClassifier(
            objective=xgb_cfg["objective"],
            tree_method=xgb_cfg["tree_method"],
            device=xgb_cfg["device"],
            eval_metric=xgb_cfg["eval_metric"],
            random_state=xgb_cfg["random_state"],
            n_jobs=config["compute"]["xgboost_n_jobs"],
        )
        stored = xgb_model.get_params()
        report["constructors_accept_config"] = True
        report["xgboost_device_effective"] = stored.get("device")
        report["xgboost_tree_method_effective"] = stored.get("tree_method")
        if stored.get("device") != "cpu" or stored.get("tree_method") != "hist":
            problems.append(
                "XGBoost n'a pas conservé device=cpu et tree_method=hist."
            )
        xgb_model.set_params(scale_pos_weight=None)
        report["xgboost_accepts_scale_pos_weight"] = (
            "scale_pos_weight" in xgb_model.get_params()
        )
        if not report["xgboost_accepts_scale_pos_weight"]:
            problems.append("XGBClassifier n'accepte pas scale_pos_weight.")
    except (TypeError, ValueError) as exc:
        report["constructors_accept_config"] = False
        problems.append(f"Un constructeur refuse la configuration CPU : {exc}")

    matplotlib.use(config["runtime"]["matplotlib_backend"], force=True)
    import matplotlib.pyplot  # noqa: F401
    import seaborn  # noqa: F401

    report["shap_tree_explainer_present"] = hasattr(shap, "TreeExplainer")
    if not hasattr(shap, "TreeExplainer"):
        problems.append("SHAP ne publie pas TreeExplainer.")
    return report


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
