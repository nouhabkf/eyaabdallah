# Évaluation finale sur le vendredi

XGBoost est utilisé tel qu'il a été enregistré. Il n'a pas été réentraîné. Les seuils 0,50 et 0,01 étaient déjà fixés : 0,50 est la référence, 0,01 est le seuil officiel choisi sur le jeudi. Aucun des chiffres ci-dessous n'a servi à modifier le modèle, le seuil ou les caractéristiques. SHAP et l'audit des variables n'ont pas été lancés.

Vendredi : 595 723 flux, dont 220 221 attaques (36.97%) et 375 502 flux normaux. Familles : Bot 1 953, DDoS 128 013, PortScan 90 255.

PR-AUC, calculée une fois sur les probabilités continues : **0.806112**.

## Les deux seuils

| | Seuil 0,50 | Seuil 0,01 |
|---|---:|---:|
| Accuracy | 0.640623 | 0.771246 |
| Précision | 0.993720 | 0.996934 |
| Rappel | 0.028022 | 0.382370 |
| F1 | 0.054507 | 0.552740 |
| FPR | 0.000104 | 0.000690 |
| FNR | 0.971978 | 0.617630 |
| TN | 375 463 | 375 243 |
| FP | 39 | 259 |
| FN | 214 050 | 136 015 |
| TP | 6 171 | 84 206 |

## Fausses alertes et attaques manquées

Au seuil 0,50, le modèle laisse passer 214 050 attaques et lève 39 fausses alertes. Au seuil officiel 0,01, les attaques manquées passent à 136 015 (-78035) et les fausses alertes à 259 (+220). Le rappel passe de 0.028022 à 0.382370. La précision monte, de 0.993720 à 0.996934, parce que les détections ajoutées sont en grande majorité de vraies attaques. La PR-AUC ne change pas.

### Par étiquette d'origine

| Étiquette | Seuil | TP | FN | FP |
|---|---:|---:|---:|---:|
| BENIGN | 0.50 | 0 | 0 | 39 |
| Bot | 0.50 | 0 | 1 953 | 0 |
| DDoS | 0.50 | 5 914 | 122 099 | 0 |
| PortScan | 0.50 | 257 | 89 998 | 0 |
| BENIGN | 0.01 | 0 | 0 | 259 |
| Bot | 0.01 | 0 | 1 953 | 0 |
| DDoS | 0.01 | 83 050 | 44 963 | 0 |
| PortScan | 0.01 | 1 156 | 89 099 | 0 |

## Temps d'inférence

La mesure entoure `predict_proba` du pipeline complet, imputation comprise, avec `time.perf_counter`. Cinq passages sont faits sur tout le vendredi. La moyenne et le temps par flux utilisent ces cinq passages. Le même chronométrage est répété sur les 10 000 premiers flux du vendredi, comme prévu dans la configuration. Ces temps ne servent à aucun choix.

Vendredi complet : 0.581 s, 0.573 s, 0.503 s, 0.477 s, 0.489 s. Moyenne 0.525 s, soit 8.808e-07 s par flux.

Lot de 10 000 flux : moyenne 0.009 s, soit 9.328e-07 s par flux.

Graphiques : `outputs/figures/friday_confusion_matrices.png`, `outputs/figures/friday_threshold_comparison.png`, `outputs/figures/friday_precision_recall.png`.

Résultats : `outputs/metrics/friday_evaluation.json` et `outputs/tables/friday_evaluation.csv`.
