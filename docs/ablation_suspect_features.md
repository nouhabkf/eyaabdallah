# Ablation des trois variables suspectes

Cette expérience réentraîne XGBoost sans Destination Port, Init_Win_bytes_backward et Init_Win_bytes_forward. Le modèle officiel, son fichier et ses résultats restent ceux déjà publiés. Le seuil n'est pas recherché à nouveau. Le vendredi n'est pas scoré. SHAP et la stabilité ne sont pas lancés.

## Protocole

Les lignes sont le train du lundi au mercredi et la validation du jeudi, aux effectifs déjà validés. L'imputation par médiane est ajustée dans le pipeline, sur ce train seulement. Les hyperparamètres sont ceux du réentraînement officiel : 100 arbres, profondeur 8, taux d'apprentissage 0,2, subsample 1, colsample_bytree 1, min_child_weight 1, `tree_method=hist`, CPU, graine 42. `scale_pos_weight` est recalculé sur les étiquettes de ce train, comme dans l'ajustement d'origine. L'arrêt anticipé reste désactivé.

Le temps mesuré entoure `fit` du pipeline complet, imputation comprise, avec `time.perf_counter`. C'est la même mesure que les 20.921 s du modèle officiel.

La PR-AUC est `average_precision_score` sur `predict_proba` de la classe attaque. Précision, rappel, F1 et matrice de confusion sont calculés aux deux seuils déjà figés, 0,50 et 0,01, avec la règle score supérieur ou égal au seuil. Le dernier groupe du fichier parquet dépasse la fin du jeudi : ces lignes supplémentaires sont écartées avant la matrice, avant l'imputation et avant tout score. Aucune métrique de vendredi n'est produite.

## Comparaison sur le jeudi

PR-AUC officielle : 0.932439. PR-AUC sans les trois variables : 0.618205. Écart : -0.314234.

| Modèle | Seuil | Précision | Rappel | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Officiel | 0,50 | 0.953207 | 0.832033 | 0.888508 | 395842 | 89 | 366 | 1813 |
| Sans les trois variables | 0,50 | 0.638876 | 0.636531 | 0.637701 | 395147 | 784 | 792 | 1387 |
| Officiel | 0,01 | 0.851127 | 0.918311 | 0.883444 | 395581 | 350 | 178 | 2001 |
| Sans les trois variables | 0,01 | 0.110832 | 0.911427 | 0.197632 | 379998 | 15933 | 193 | 1986 |

Au seuil officiel 0,01, les fausses alertes passent de 350 à 15933 (+15583) et les attaques manquées de 178 à 193 (+15).

Entraînement officiel : 20.921 s. Entraînement sans les trois variables : 21.846 s.

## Lecture

Le retrait enlève une part importante du classement sur le jeudi : la PR-AUC passe de 0,932 à 0,618. Cette baisse ne dépend pas du seuil. Au seuil 0,50, le F1 passe de 0,889 à 0,638 : les fausses alertes montent de 89 à 784 et les attaques manquées de 366 à 792. Au seuil 0,01, déjà choisi pour l'autre modèle, le rappel reste proche, 0,911 contre 0,918, mais la précision tombe de 0,851 à 0,111. Les fausses alertes passent de 350 à 15 933, soit un taux de 4,02 % du trafic normal contre 0,088 % auparavant. Les attaques manquées ne bougent que de 178 à 193.

Le jeudi est fait d'attaques web dirigées vers un seul port, avec une fenêtre TCP très répétée. La chute mesure ce que ces trois raccourcis apportaient sur cette validation. Elle ne décrit pas le comportement sur d'autres familles. Le seuil n'a pas été recalculé pour la nouvelle distribution de scores : une partie de l'explosion des alertes à 0,01 vient de ce choix figé, tandis que la baisse de PR-AUC vient du classement lui-même.

Le temps d'entraînement reste du même ordre, 21,8 s contre 20,9 s. Le dernier groupe parquet déborde de 24 757 lignes sur le vendredi ; ces lignes ont été écartées avant la matrice. Aucune n'a été scorée.

Ce modèle ne remplace pas le XGBoost officiel. Le seuil 0,01 n'a pas été réoptimisé pour la nouvelle distribution de scores.

## Fichiers

- `outputs/metrics/ablation_suspect_features.json`
- `outputs/tables/ablation_suspect_features.csv`
- `outputs/models/xgboost_without_suspect_features.joblib`
- `outputs/metrics/validation_scores_xgboost_without_suspects.npy`
