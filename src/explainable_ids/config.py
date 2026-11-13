"""Chargement et validation de la configuration.

La validation encode le protocole accepté. Elle ne lit aucune donnée
et n'entraîne aucun modèle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
APPROVED_DATA_ROOT = Path("C:/data/cic-ids2017")
APPROVED_VENV = Path("C:/venvs/explainable-ids")

EXPECTED_PROTOCOL_ORDER = [
    "clean",
    "exact_deduplicate",
    "main_split",
    "tune_on_train_subsample",
    "refit_on_full_train",
    "select_primary_model_by_validation_pr_auc",
    "select_threshold_on_validation",
    "evaluate_test_once",
    "explain_without_changing_model",
    "audit_without_test_decisions",
    "stability_independent_partitions",
]

REQUIRED_PARTITION_FIELDS = {
    "experiment_id",
    "experiment_type",
    "split_seed",
    "model_seed",
    "stratify",
    "train_percent",
    "validation_percent",
    "test_percent",
    "train_indices_sha256",
    "validation_indices_sha256",
    "test_indices_sha256",
    "n_train",
    "n_validation",
    "n_test",
    "hyperparameters_frozen_from",
    "created_utc",
}

SHAP_CASE_IDS = {
    "tp_confident",
    "tp_marginal",
    "fp_confident",
    "fp_marginal",
    "fn_marginal",
}

STILL_BLOCKED_PHASES: tuple[str, ...] = ()


class ConfigError(Exception):
    """La configuration viole une règle du protocole."""


def load_yaml(path: Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration illisible : {config_path}")
    return loaded


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Charge le YAML et refuse une configuration qui viole le protocole."""
    cfg = load_yaml(path)
    errors = validate_config(cfg)
    if errors:
        joined = "\n".join(f"- {item}" for item in errors)
        raise ConfigError(f"Configuration refusée :\n{joined}")
    return cfg


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Retourne les violations. Une liste vide signifie une configuration acceptable."""
    if not isinstance(cfg, dict):
        return ["La configuration doit être un objet."]

    errors: list[str] = []
    _check_phases(cfg, errors)
    _check_paths(cfg, errors)
    _check_dataset_phase(cfg, errors)
    _check_labels(cfg, errors)
    _check_cleaning(cfg, errors)
    _check_split(cfg, errors)
    _check_chronological(cfg, errors)
    _check_imbalance(cfg, errors)
    _check_models_and_search(cfg, errors)
    _check_threshold_and_metrics(cfg, errors)
    _check_explanations(cfg, errors)
    _check_feature_decisions(cfg, errors)
    _check_stability(cfg, errors)
    _check_compute(cfg, errors)
    _check_protocol_order(cfg, errors)
    _reject_gpu_and_smote_values(cfg, errors)
    return errors


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Résout les dossiers approuvés. Les données restent hors du dépôt."""
    data_root = Path(cfg["paths"]["data_root"])
    return {
        "project_root": PROJECT_ROOT,
        "data_root": data_root,
        "raw": data_root / cfg["paths"]["raw_dirname"],
        "interim": data_root / cfg["paths"]["interim_dirname"],
        "processed": data_root / cfg["paths"]["processed_dirname"],
        "splits": data_root / cfg["paths"]["splits_dirname"],
        "stability_splits": data_root / cfg["paths"]["splits_dirname"] / "stability",
        "chronological_splits": data_root
        / cfg["paths"]["splits_dirname"]
        / "chronological",
        "grouped_splits": data_root / cfg["paths"]["splits_dirname"] / "grouped",
        "outputs": PROJECT_ROOT / cfg["paths"]["outputs_dirname"],
        "metrics": PROJECT_ROOT / cfg["paths"]["metrics_dirname"],
        "figures": PROJECT_ROOT / cfg["paths"]["figures_dirname"],
        "tables": PROJECT_ROOT / cfg["paths"]["tables_dirname"],
        "models": PROJECT_ROOT / cfg["paths"]["models_dirname"],
        "venv": Path(cfg["runtime"]["venv"]),
    }


def _get(cfg: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = cfg
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _need(
    cfg: dict[str, Any],
    path: str,
    errors: list[str],
    expected_type: type | tuple[type, ...] | None = None,
) -> Any:
    found, value = _get(cfg, path)
    if not found:
        errors.append(f"Clé manquante : {path}")
        return None
    if expected_type is not None and not isinstance(value, expected_type):
        errors.append(
            f"Type inattendu pour {path} : {type(value).__name__}."
        )
        return None
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_phases(cfg: dict[str, Any], errors: list[str]) -> None:
    phases = _need(cfg, "phases", errors, dict)
    if phases is None:
        return
    architecture = phases.get("architecture")
    if architecture != "open":
        errors.append("La phase architecture doit être ouverte.")
    if phases.get("data_preparation") != "open":
        errors.append("La préparation des données validée doit être ouverte.")
    if phases.get("training") != "open":
        errors.append("La phase training autorisée doit être ouverte.")
    if phases.get("evaluation") != "open":
        errors.append("La phase evaluation autorisée doit être ouverte.")
    for name in STILL_BLOCKED_PHASES:
        if phases.get(name) != "blocked":
            errors.append(
                f"La phase {name} doit rester bloquée : l'audit des variables n'est pas autorisé."
            )
    for name, state in phases.items():
        if state not in {"open", "blocked"}:
            errors.append(f"État de phase inconnu pour {name} : {state}.")


def _check_paths(cfg: dict[str, Any], errors: list[str]) -> None:
    data_root = _need(cfg, "paths.data_root", errors, str)
    venv = _need(cfg, "runtime.venv", errors, str)
    if data_root is not None:
        resolved = Path(data_root)
        if resolved.resolve() != APPROVED_DATA_ROOT.resolve():
            errors.append(
                f"Les données doivent être stockées dans {APPROVED_DATA_ROOT}."
            )
        if "onedrive" in str(resolved).lower():
            errors.append("Le dossier de données est sous OneDrive.")
    if venv is not None:
        resolved_venv = Path(venv)
        if resolved_venv.resolve() != APPROVED_VENV.resolve():
            errors.append(f"L'environnement virtuel attendu est {APPROVED_VENV}.")
        if "onedrive" in str(resolved_venv).lower():
            errors.append("L'environnement virtuel est sous OneDrive.")
    path_map = _need(cfg, "paths", errors, dict)
    if path_map is None:
        return
    for key, value in path_map.items():
        if key == "data_root":
            continue
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            errors.append(f"Chemin relatif invalide : paths.{key}.")


def _check_dataset_phase(cfg: dict[str, Any], errors: list[str]) -> None:
    status = _need(cfg, "dataset.local_status", errors, str)
    download = _need(cfg, "dataset.download_during_architecture_phase", errors, bool)
    is_filter = _need(cfg, "dataset.expected_filenames_are_a_filter", errors, bool)
    names = _need(cfg, "dataset.expected_filenames_to_confirm", errors, list)
    if status != "present":
        errors.append("Les CSV validés doivent être marqués présents, sans être relus comme une source nouvelle.")
    if download is True:
        errors.append("Le téléchargement du dataset est interdit dans cette phase.")
    if is_filter is True:
        errors.append("La liste de fichiers attendus ne doit pas servir de filtre.")
    if isinstance(names, list) and not names:
        errors.append("La liste des fichiers à confirmer est vide.")


def _check_labels(cfg: dict[str, Any], errors: list[str]) -> None:
    mapping = _need(cfg, "labels.mapping", errors, str)
    is_filter = _need(cfg, "labels.known_labels_are_a_filter", errors, bool)
    positive = _need(cfg, "labels.positive_class_value", errors)
    negative = _need(cfg, "labels.negative_class_value", errors)
    if mapping != "all_non_benign_are_attacks":
        errors.append("Toute étiquette non bénigne doit être une attaque.")
    if is_filter is True:
        errors.append("Une liste fermée d'attaques ne doit pas filtrer les étiquettes.")
    if positive != 1 or negative != 0:
        errors.append("L'attaque doit être codée 1 et le trafic normal 0.")


def _check_cleaning(cfg: dict[str, Any], errors: list[str]) -> None:
    order = _need(cfg, "cleaning.order", errors, list)
    if isinstance(order, list):
        if any("impute" in str(step) for step in order):
            errors.append("L'imputation est interdite avant le découpage.")
        if "drop_exact_duplicate_rows" not in order:
            errors.append("La suppression des doublons exacts doit précéder le découpage.")
    before = _need(cfg, "cleaning.deduplicate.before_split", errors, bool)
    if before is False:
        errors.append("Les doublons exacts doivent être supprimés avant le découpage.")
    if _need(cfg, "cleaning.drop_rows_with_missing_features", errors, bool) is True:
        errors.append(
            "Les lignes incomplètes ne sont pas supprimées en masse : l'imputation médiane est ajustée sur le train."
        )
    fit_on = _need(cfg, "train_only_feature_filters.fit_on", errors, str)
    forbidden = _need(cfg, "train_only_feature_filters.forbidden_fit_on", errors, list)
    if fit_on != "train":
        errors.append("Les filtres de variables s'ajustent sur le train.")
    if isinstance(forbidden, list) and (
        "validation" not in forbidden or "test" not in forbidden
    ):
        errors.append("Les filtres de variables ne s'ajustent ni sur la validation ni sur le test.")
    if _need(cfg, "train_only_feature_filters.scale_for_trees", errors, str) != "none":
        errors.append("Les arbres ne sont pas standardisés.")
    if _need(cfg, "train_only_feature_filters.drop_near_zero_variance", errors, bool) is True:
        errors.append(
            "Une variable à faible variance n'est pas supprimée sans justification d'audit."
        )
    if _need(cfg, "columns.timestamp_use", errors, str) != "chronological_analysis_only":
        errors.append("L'horodatage ne peut pas servir de variable prédictive.")


def _check_split(cfg: dict[str, Any], errors: list[str]) -> None:
    train = _need(cfg, "split.train_percent", errors)
    validation = _need(cfg, "split.validation_percent", errors)
    test = _need(cfg, "split.test_percent", errors)
    if _is_int(train) and _is_int(validation) and _is_int(test):
        if (train, validation, test) != (70, 15, 15):
            errors.append("Le découpage groupé complémentaire doit être 70 / 15 / 15.")
    else:
        errors.append("Les parts du découpage doivent être des entiers.")
    if _need(cfg, "split.role", errors, str) != "complementary_grouped_stratified":
        errors.append("Le découpage 70 / 15 / 15 est le protocole groupé complémentaire.")
    if _need(cfg, "split.stratify", errors, bool) is not True:
        errors.append("Le découpage groupé complémentaire doit être stratifié.")
    for key in (
        "selects_official_model",
        "selects_official_threshold",
        "selects_official_features",
        "selects_official_hyperparameters",
    ):
        if _need(cfg, f"split.{key}", errors, bool) is not False:
            errors.append(f"Le découpage groupé ne décide pas le protocole officiel : {key}.")
    if _need(cfg, "split.deduplicate_exact_before_split", errors, bool) is not True:
        errors.append("Le découpage doit être précédé de la suppression des doublons exacts.")
    seed = _need(cfg, "split.seed", errors)
    if not _is_int(seed):
        errors.append("La graine du découpage principal doit être un entier.")
    limitations = _need(cfg, "split.limitations", errors, list)
    if isinstance(limitations, list) and len(limitations) < 3:
        errors.append("Les limites du découpage aléatoire doivent être documentées.")


def _check_chronological(cfg: dict[str, Any], errors: list[str]) -> None:
    section = _need(cfg, "chronological_analysis", errors, dict)
    if not isinstance(section, dict):
        return
    if section.get("role") != "official_evaluation":
        errors.append("Le découpage chronologique est l'évaluation officielle.")
    expected_days = {
        "train_days": ["Monday", "Tuesday", "Wednesday"],
        "validation_days": ["Thursday-Morning", "Thursday-Afternoon"],
        "test_days": [
            "Friday-Morning",
            "Friday-Afternoon-PortScan",
            "Friday-Afternoon-DDoS",
        ],
    }
    for key, expected in expected_days.items():
        if section.get(key) != expected:
            errors.append(f"Les jours officiels de {key} ont été modifiés.")
    if section.get("validation_selects_model") is not True:
        errors.append("La validation chronologique choisit le modèle officiel.")
    if section.get("validation_selects_threshold") is not True:
        errors.append("La validation chronologique choisit le seuil officiel.")
    for key in (
        "test_selects_model",
        "test_selects_threshold",
        "test_selects_features",
        "test_selects_hyperparameters",
    ):
        if section.get(key) is not False:
            errors.append(f"Le test chronologique ne doit pas décider : {key}.")


def _check_imbalance(cfg: dict[str, Any], errors: list[str]) -> None:
    if _need(cfg, "imbalance.smote", errors, bool) is not False:
        errors.append("SMOTE est interdit.")
    if _need(cfg, "imbalance.synthetic_samples", errors, bool) is not False:
        errors.append("Les échantillons synthétiques sont interdits.")
    if _need(cfg, "imbalance.logistic_regression", errors, str) != "class_weight":
        errors.append("La régression logistique utilise class_weight.")
    if _need(cfg, "imbalance.random_forest", errors, str) != "class_weight":
        errors.append("La forêt aléatoire utilise class_weight.")
    if _need(cfg, "imbalance.xgboost", errors, str) != "scale_pos_weight":
        errors.append("XGBoost utilise scale_pos_weight calculé sur le train.")
    mode = _need(cfg, "models.xgboost.scale_pos_weight.mode", errors, str)
    fit_on = _need(cfg, "models.xgboost.scale_pos_weight.fit_on", errors, str)
    forbid_constant = _need(
        cfg, "models.xgboost.scale_pos_weight.forbid_precomputed_constant", errors, bool
    )
    if mode != "negative_count_over_positive_count" or fit_on != "train":
        errors.append("scale_pos_weight se calcule par n_normal / n_attack sur le train.")
    if forbid_constant is not True:
        errors.append("Une constante scale_pos_weight précalculée est interdite.")
    if _need(cfg, "models.logistic_regression.class_weight", errors, str) != "balanced":
        errors.append("class_weight de la régression logistique doit être balanced.")
    if _need(cfg, "models.random_forest.class_weight", errors, str) != "balanced":
        errors.append("class_weight de la forêt aléatoire doit être balanced.")


def _check_models_and_search(cfg: dict[str, Any], errors: list[str]) -> None:
    if _need(cfg, "hyperparameters.test_access", errors, bool) is not False:
        errors.append("La recherche d'hyperparamètres n'a pas accès au test.")
    if _need(cfg, "hyperparameters.validation_access_during_search", errors, bool) is not False:
        errors.append("La recherche d'hyperparamètres n'utilise pas la validation officielle.")
    if _need(cfg, "hyperparameters.early_stopping.uses_official_validation", errors, bool) is not False:
        errors.append("L'arrêt anticipé ne doit pas consommer la validation officielle.")
    if _need(cfg, "hyperparameters.early_stopping.mode", errors, str) != "none":
        errors.append("Le nombre d'arbres est un hyperparamètre. L'arrêt anticipé est désactivé.")
    if _need(cfg, "hyperparameters.refit_winner_on", errors, str) != "full_train":
        errors.append("Le modèle retenu est réentraîné sur tout le train.")
    if _need(cfg, "hyperparameters.subsample.source", errors, str) != "train":
        errors.append("Le sous-échantillon de recherche est tiré dans le train.")
    max_rows = _need(cfg, "hyperparameters.subsample.max_rows", errors)
    hard_cap = _need(cfg, "hyperparameters.subsample.hard_cap_rows", errors)
    if not _is_int(max_rows) or not _is_int(hard_cap) or max_rows > hard_cap:
        errors.append("Le sous-échantillon de recherche dépasse le plafond autorisé.")
    if _need(cfg, "hyperparameters.subsample.stratify", errors, bool) is not True:
        errors.append("Le sous-échantillon de recherche doit être stratifié.")
    scoring = _need(cfg, "hyperparameters.inner_cv.scoring", errors, str)
    scoring_input = _need(cfg, "hyperparameters.inner_cv.scoring_input", errors, str)
    if scoring != "average_precision" or scoring_input != "continuous_scores":
        errors.append("La recherche optimise la PR-AUC à partir des scores continus.")
    cap = _need(cfg, "compute.max_search_iterations", errors)
    for family in ("random_forest", "xgboost"):
        n_iter = _need(cfg, f"hyperparameters.{family}.n_iter", errors)
        if not _is_int(n_iter) or not _is_int(cap) or n_iter > cap:
            errors.append(f"n_iter de {family} dépasse le plafond CPU.")
    if _need(cfg, "model_selection.split", errors, str) != "validation":
        errors.append("Le modèle principal se choisit sur la validation.")
    if _need(cfg, "model_selection.input", errors, str) != "continuous_scores":
        errors.append("Le choix du modèle utilise les scores continus.")
    if _need(cfg, "model_selection.metric", errors, str) != "average_precision":
        errors.append("Le choix du modèle utilise la PR-AUC.")
    if _need(cfg, "models.xgboost.device", errors, str) != "cpu":
        errors.append("XGBoost doit rester sur CPU.")
    if _need(cfg, "models.xgboost.tree_method", errors, str) != "hist":
        errors.append("XGBoost utilise tree_method hist.")
    if _need(cfg, "models.xgboost.eval_metric_role", errors, str) != "internal_training_only":
        errors.append("La métrique interne XGBoost n'est pas la PR-AUC publiée.")


def _check_threshold_and_metrics(cfg: dict[str, Any], errors: list[str]) -> None:
    if _need(cfg, "threshold.selection_split", errors, str) != "validation":
        errors.append("Le seuil se choisit exclusivement sur la validation.")
    forbidden = _need(cfg, "threshold.forbidden_splits", errors, list)
    if isinstance(forbidden, list) and (
        "test" not in forbidden or "train" not in forbidden
    ):
        errors.append("Le seuil ne se choisit ni sur le train ni sur le test.")
    if _need(cfg, "threshold.objective", errors, str) != "weighted_fpr_fnr":
        errors.append("Le seuil minimise le coût pondéré du FPR et du FNR.")
    fp_weight = _need(cfg, "threshold.fp_weight", errors)
    fn_weight = _need(cfg, "threshold.fn_weight", errors)
    if not isinstance(fp_weight, (int, float)) or not isinstance(fn_weight, (int, float)):
        errors.append("Les poids du seuil doivent être numériques.")
    elif isinstance(fp_weight, bool) or isinstance(fn_weight, bool) or fp_weight <= 0 or fn_weight <= 0:
        errors.append("Les poids du seuil doivent être strictement positifs.")
    if _need(cfg, "threshold.weights_locked", errors, bool) is not True:
        errors.append("Les poids du compromis faux positifs / faux négatifs sont figés.")
    grid = _need(cfg, "threshold.grid", errors, dict)
    if isinstance(grid, dict):
        start, stop, step = grid.get("start"), grid.get("stop"), grid.get("step")
        if not all(_is_int(item) for item in (start, stop, step)):
            errors.append("La grille de seuils est exprimée en centièmes entiers.")
        elif not (1 <= start <= stop <= 99 and step >= 1 and 50 in range(start, stop + 1, step)):
            errors.append("La grille de seuils doit contenir 0,50.")
    fixed = _need(cfg, "threshold.always_evaluate_hundredths", errors, list)
    if isinstance(fixed, list) and 50 not in fixed:
        errors.append("Le seuil 0,5 doit aussi être évalué.")
    if _need(cfg, "threshold.score_source", errors, str) != "predict_proba_positive_class":
        errors.append("Le seuil s'applique aux scores continus de la classe attaque.")
    published = _need(cfg, "metrics.published_pr_auc.input", errors, str)
    forbid_input = _need(cfg, "metrics.published_pr_auc.forbid_input", errors, str)
    if published != "continuous_scores" or forbid_input != "binary_predictions":
        errors.append("La PR-AUC publiée se calcule sur les scores continus.")
    if _need(cfg, "metrics.internal_eval_is_the_published_metric", errors, bool) is not False:
        errors.append("aucpr interne à XGBoost ne remplace pas average_precision_score.")
    if _need(cfg, "metrics.accuracy_is_not_a_selection_metric", errors, bool) is not True:
        errors.append("L'accuracy ne sert pas à choisir le modèle.")
    if _need(cfg, "scores.source", errors, str) != "predict_proba_positive_class":
        errors.append("Les scores publiés viennent de predict_proba de la classe positive.")
    if _need(cfg, "evaluation.official_test_reads", errors, str) != "single_final_pass":
        errors.append("Le test officiel n'est lu qu'une fois.")
    forbid_scores = _need(cfg, "scores.forbid", errors, list)
    if isinstance(forbid_scores, list) and "binary_predictions" not in forbid_scores:
        errors.append("Les prédictions binaires ne remplacent pas les scores continus.")


def _check_explanations(cfg: dict[str, Any], errors: list[str]) -> None:
    if _need(cfg, "explanations.permutation.split", errors, str) != "validation":
        errors.append("L'importance par permutation utilisée pour décider se calcule sur la validation.")
    if _need(cfg, "explanations.shap_global.split", errors, str) != "validation":
        errors.append("Le SHAP global utilisé pour décider se calcule sur la validation.")
    if _need(cfg, "explanations.shap_global.explainer", errors, str) != "tree":
        errors.append("L'explicateur prévu est TreeExplainer.")
    if _need(cfg, "explanations.shap_global.forbid_kernel_explainer", errors, bool) is not True:
        errors.append("KernelSHAP est exclu pour des raisons de coût CPU.")
    _check_row_cap(
        cfg,
        errors,
        "explanations.shap_global.explained_rows",
        "explanations.shap_global.hard_cap_rows",
        "SHAP global",
    )
    _check_row_cap(
        cfg,
        errors,
        "explanations.permutation.max_rows",
        "explanations.permutation.hard_cap_rows",
        "permutation",
    )
    repeats = _need(cfg, "explanations.permutation.n_repeats", errors)
    repeat_cap = _need(cfg, "explanations.permutation.hard_cap_repeats", errors)
    if not _is_int(repeats) or not _is_int(repeat_cap) or repeats > repeat_cap:
        errors.append("Le nombre de répétitions de permutation dépasse le plafond CPU.")

    if _need(cfg, "shap_cases.selection_uses_shap_values", errors, bool) is not False:
        errors.append("Les cinq cas SHAP se choisissent avant le calcul de leurs explications.")
    if _need(cfg, "shap_cases.explanation_computed_after_selection", errors, bool) is not True:
        errors.append("L'explication locale vient après la sélection des cas.")
    if _need(cfg, "shap_cases.rescore_allowed", errors, bool) is not False:
        errors.append("La sélection des cas réutilise les scores du test officiel, sans nouveau passage.")
    forbidden_uses = _need(cfg, "shap_cases.forbidden_uses", errors, list)
    required_forbidden = {
        "feature_selection",
        "suspect_feature_decision",
        "threshold_selection",
        "hyperparameter_selection",
        "model_selection",
    }
    if isinstance(forbidden_uses, list) and not required_forbidden.issubset(forbidden_uses):
        errors.append("Les cas SHAP locaux ne peuvent servir à aucune décision de modèle.")
    rules = _need(cfg, "shap_cases.rules", errors, list)
    if isinstance(rules, list):
        identifiers = [rule.get("id") for rule in rules if isinstance(rule, dict)]
        counts = [rule.get("count") for rule in rules if isinstance(rule, dict)]
        if set(identifiers) != SHAP_CASE_IDS or counts != [1, 1, 1, 1, 1]:
            errors.append(
                "La règle fixe deux vrais positifs, deux faux positifs et un faux négatif."
            )
    if _need(cfg, "shap_cases.empty_category_policy", errors, str) != "document_absence_do_not_invent":
        errors.append("Une catégorie de cas absente est documentée, pas remplacée par un exemple inventé.")
    if _need(cfg, "shap_cases.fallback", errors) not in {None, "none"}:
        errors.append("Aucun cas de repli n'est autorisé après avoir vu les explications.")


def _check_row_cap(
    cfg: dict[str, Any],
    errors: list[str],
    rows_path: str,
    cap_path: str,
    label: str,
) -> None:
    rows = _need(cfg, rows_path, errors)
    cap = _need(cfg, cap_path, errors)
    if not _is_int(rows) or not _is_int(cap) or rows > cap:
        errors.append(f"La taille de {label} dépasse le plafond CPU.")


def _check_feature_decisions(cfg: dict[str, Any], errors: list[str]) -> None:
    allowed = _need(cfg, "feature_decisions.allowed_evidence_splits", errors, list)
    forbidden = _need(cfg, "feature_decisions.forbidden_evidence_splits", errors, list)
    if isinstance(allowed, list) and set(allowed) != {"train", "validation"}:
        errors.append("Les décisions de variables n'utilisent que le train et la validation.")
    if isinstance(forbidden, list) and "test" not in forbidden:
        errors.append("Le test ne décide jamais du retrait d'une variable.")
    if _need(cfg, "feature_decisions.suspect_feature.test_metrics_may_decide", errors, bool) is not False:
        errors.append("Une métrique de test ne peut pas désigner une variable suspecte.")
    if _need(cfg, "feature_decisions.suspect_feature.ablation_replaces_official_score", errors, bool) is not False:
        errors.append("L'ablation ne remplace pas le score officiel.")
    if _need(cfg, "feature_decisions.suspect_feature.domain_reason_required", errors, bool) is not True:
        errors.append("Une variable suspecte exige une raison de domaine écrite avant l'ablation.")
    if _need(cfg, "feature_decisions.suspect_feature.minimum_methods_in_top_k", errors) != 2:
        errors.append("Une variable suspecte figure dans le top 10 d'au moins deux méthodes.")
    if (
        _need(cfg, "feature_decisions.suspect_feature.if_no_candidate", errors, str)
        != "report_none_and_do_not_force_ablation"
    ):
        errors.append("L'absence de variable suspecte est un résultat acceptable.")


def _check_stability(cfg: dict[str, Any], errors: list[str]) -> None:
    if _need(cfg, "stability.select_best_run_as_official", errors, bool) is not False:
        errors.append("La meilleure répétition de stabilité ne devient pas le résultat officiel.")
    if _need(cfg, "stability.reads_official_test_to_choose_a_winner", errors, bool) is not False:
        errors.append("Le test officiel ne sert pas à choisir une répétition de stabilité.")
    if _need(cfg, "stability.not_an_external_validation", errors, bool) is not True:
        errors.append("La stabilité doit être décrite comme une sensibilité, pas comme une validation externe.")
    if _need(cfg, "stability.pass_fail_cutoff", errors) is not None:
        errors.append("Aucun seuil de réussite de stabilité n'est fixé après coup.")
    if _need(cfg, "stability.never_recompute_from_partition_test", errors, bool) is not True:
        errors.append("Le test d'une partition de stabilité ne recalcule ni seuil ni poids.")
    if _need(cfg, "stability.split_seed_experiments.independent_partitions", errors, bool) is not True:
        errors.append("Les graines de découpage produisent des partitions indépendantes.")
    if _need(cfg, "stability.split_seed_experiments.must_not_copy_official_indices", errors, bool) is not True:
        errors.append("Les partitions de stabilité ne recopient pas les indices officiels.")
    if _need(cfg, "stability.model_seed_experiments.partition", errors, str) != "official":
        errors.append("Les graines de modèle partagent la partition officielle, afin de ne faire varier que la graine.")
    official_seed = _need(cfg, "split.seed", errors)
    split_seeds = _need(cfg, "stability.split_seed_experiments.seeds", errors, list)
    model_seeds = _need(cfg, "stability.model_seed_experiments.seeds", errors, list)
    if isinstance(split_seeds, list):
        if not all(_is_int(seed) for seed in split_seeds) or len(split_seeds) != len(set(split_seeds)):
            errors.append("Les graines de découpage de stabilité doivent être des entiers distincts.")
        elif official_seed in split_seeds:
            errors.append("Les partitions de stabilité utilisent d'autres graines que le découpage officiel.")
        if len(split_seeds) != 3:
            errors.append("Trois partitions indépendantes sont prévues.")
    if isinstance(model_seeds, list):
        if not all(_is_int(seed) for seed in model_seeds) or len(model_seeds) != len(set(model_seeds)):
            errors.append("Les graines de modèle doivent être des entiers distincts.")
        if len(model_seeds) != 3:
            errors.append("Trois graines de modèle sont prévues.")
    fields = _need(cfg, "stability.partition_record.required_fields", errors, list)
    if isinstance(fields, list) and not REQUIRED_PARTITION_FIELDS.issubset(fields):
        errors.append("Chaque partition de stabilité doit enregistrer les champs de traçabilité prévus.")
    recompute_train = _need(cfg, "stability.recompute_on_each_partition_train", errors, list)
    if isinstance(recompute_train, list) and "scale_pos_weight" not in recompute_train:
        errors.append("scale_pos_weight est recalculé sur le train de chaque partition.")
    recompute_val = _need(cfg, "stability.recompute_on_each_partition_validation", errors, list)
    if isinstance(recompute_val, list) and recompute_val != ["threshold"]:
        errors.append("Sur chaque partition, seule la validation recalcule le seuil.")


def _check_compute(cfg: dict[str, Any], errors: list[str]) -> None:
    if _need(cfg, "compute.device", errors, str) != "cpu":
        errors.append("Le calcul autorisé est CPU.")
    if _need(cfg, "compute.allow_gpu", errors, bool) is not False:
        errors.append("Le GPU est interdit.")
    if _need(cfg, "compute.xgboost_device", errors, str) != "cpu":
        errors.append("Le device XGBoost doit être cpu.")
    if _need(cfg, "compute.pandas_numeric_dtype", errors, str) != "float32":
        errors.append("Les variables numériques sont stockées en float32.")
    jobs = _need(cfg, "compute.sklearn_n_jobs", errors)
    xgb_jobs = _need(cfg, "compute.xgboost_n_jobs", errors)
    if not _is_int(jobs) or not _is_int(xgb_jobs) or jobs < 1 or xgb_jobs < 1:
        errors.append("Le nombre de processus CPU doit être un entier positif.")
    if _need(cfg, "runtime.python", errors, str) != "3.12":
        errors.append("L'interpréteur du projet est Python 3.12.")
    if _need(cfg, "runtime.pythonhashseed", errors, str) != "42":
        errors.append("PYTHONHASHSEED doit être fixé à 42.")


def _check_protocol_order(cfg: dict[str, Any], errors: list[str]) -> None:
    order = _need(cfg, "protocol_order", errors, list)
    if order != EXPECTED_PROTOCOL_ORDER:
        errors.append("L'ordre du protocole a été modifié.")


def _reject_gpu_and_smote_values(cfg: dict[str, Any], errors: list[str]) -> None:
    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key).lower() == "smote" and value is not False:
                    errors.append(f"SMOTE est interdit ({child}).")
                if isinstance(value, str) and value.lower() == "smote":
                    errors.append(f"SMOTE est interdit ({child}).")
                if isinstance(value, str) and value.lower() in {"cuda", "gpu", "gpu_hist"}:
                    errors.append(f"Calcul GPU interdit ({child} = {value}).")
                walk(value, child)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(cfg, "")
