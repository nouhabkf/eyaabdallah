"""Pipelines non ajustés. Responsable : personne B.

Régression logistique : imputation médiane, puis standardisation.
Forêt et XGBoost : imputation médiane, sans standardisation.
``fit`` n'est pas appelé ici. Lorsqu'il le sera, il ne recevra que le train.
``scale_pos_weight`` n'est pas inscrit : il sera calculé sur le train.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median")


def build_logistic_pipeline(config: dict[str, Any]) -> Pipeline:
    """Pipeline de la régression logistique, non ajusté."""
    model = config["models"]["logistic_regression"]
    return Pipeline(
        [
            ("imputer", _imputer()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=model["class_weight"],
                    solver=model["solver"],
                    max_iter=int(model["max_iter"]),
                    random_state=int(model["random_state"]),
                ),
            ),
        ]
    )


def build_random_forest_pipeline(config: dict[str, Any]) -> Pipeline:
    """Pipeline de la forêt aléatoire, non ajusté."""
    model = config["models"]["random_forest"]
    return Pipeline(
        [
            ("imputer", _imputer()),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight=model["class_weight"],
                    random_state=int(model["random_state"]),
                    n_jobs=int(config["compute"]["sklearn_n_jobs"]),
                ),
            ),
        ]
    )


def build_xgboost_pipeline(config: dict[str, Any]) -> Pipeline:
    """Pipeline XGBoost CPU, non ajusté."""
    model = config["models"]["xgboost"]
    if model["device"] != "cpu" or model["tree_method"] != "hist":
        raise RuntimeError("XGBoost doit rester sur CPU avec tree_method hist.")
    classifier = XGBClassifier(
        objective=model["objective"],
        tree_method=model["tree_method"],
        device="cpu",
        eval_metric=model["eval_metric"],
        random_state=int(model["random_state"]),
        n_jobs=int(config["compute"]["xgboost_n_jobs"]),
    )
    return Pipeline(
        [
            ("imputer", _imputer()),
            ("classifier", classifier),
        ]
    )


def build_tree_pipeline(config: dict[str, Any], family: str = "random_forest") -> Pipeline:
    """Pipeline d'un arbre, sans standardisation."""
    if family == "random_forest":
        return build_random_forest_pipeline(config)
    if family == "xgboost":
        return build_xgboost_pipeline(config)
    raise RuntimeError(f"Famille d'arbre inconnue : {family}.")
