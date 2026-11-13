# Comparaison des trois importances

Cette lecture est postérieure à l'évaluation finale. Elle ne réentraîne rien, ne change aucun hyperparamètre et ne change pas le seuil 0,01. Elle ne sélectionne aucune variable. Les fichiers de l'évaluation du vendredi ne sont pas réécrits.

## Méthode

L'impureté est celle de la forêt officielle déjà entraînée, lue dans `outputs/models/random_forest.joblib`, sur les 61 caractéristiques. Elle ne dépend pas du vendredi.

La permutation porte sur les 595723 flux du vendredi, soit tout le test. La métrique est la PR-AUC, `average_precision_score` sur `predict_proba` de la classe attaque. Chaque variable est mélangée 5 fois, graine 42, avec `n_jobs=1`. L'importance est la baisse moyenne de PR-AUC. Le modèle est le XGBoost officiel, imputation comprise, sans nouvel ajustement. Le calcul a duré 521.8 s. La PR-AUC de départ, 0.806112, reproduit celle déjà publiée, 0.806112.

Le SHAP est celui déjà calculé sur 2000 flux du vendredi, graine 42. Le classement est la moyenne des valeurs absolues, en log-odds de l'attaque. Il n'est pas recalculé.

Une prédiction sur tout le vendredi avait déjà été chronométrée à environ une demi-seconde. Cinq répétitions sur 61 variables font 306 prédictions. Le calcul mesuré a duré 521,8 s. Ce coût permettait d'utiliser le test entier, sans sous-échantillon.

## Dix premières variables

| Rang | Forêt, impureté | Permutation, vendredi | SHAP, moyenne absolue |
|---:|---|---|---|
| 1 | Packet Length Variance | Destination Port | Destination Port |
| 2 | Packet Length Std | Bwd Packet Length Mean | Bwd Packet Length Std |
| 3 | Bwd Packet Length Std | Bwd Packet Length Std | Init_Win_bytes_backward |
| 4 | Bwd Packet Length Max | Fwd IAT Min | Bwd Packet Length Mean |
| 5 | Destination Port | Bwd Packet Length Max | Init_Win_bytes_forward |
| 6 | Bwd Packet Length Mean | PSH Flag Count | Bwd Packets/s |
| 7 | Max Packet Length | Flow Duration | Fwd IAT Min |
| 8 | Packet Length Mean | Fwd IAT Total | Bwd Packet Length Min |
| 9 | Total Length of Bwd Packets | URG Flag Count | min_seg_size_forward |
| 10 | Average Packet Size | Bwd Packet Length Min | Bwd Packet Length Max |

## Rangs des variables présentes dans au moins un top 10

| Variable | Forêt | Permutation vendredi | SHAP vendredi |
|---|---:|---:|---:|
| Destination Port | 5 | 1 | 1 |
| Bwd Packet Length Mean | 6 | 2 | 4 |
| Bwd Packet Length Std | 3 | 3 | 2 |
| Fwd IAT Min | 28 | 4 | 7 |
| Bwd Packet Length Max | 4 | 5 | 10 |
| PSH Flag Count | 57 | 6 | 13 |
| Flow Duration | 32 | 7 | 24 |
| Fwd IAT Total | 17 | 8 | 20 |
| URG Flag Count | 50 | 9 | 21 |
| Bwd Packet Length Min | 35 | 10 | 8 |
| Init_Win_bytes_backward | 16 | 17 | 3 |
| Init_Win_bytes_forward | 23 | 18 | 5 |
| min_seg_size_forward | 36 | 26 | 9 |
| Total Length of Bwd Packets | 9 | 30 | 28 |
| Packet Length Variance | 1 | 35 | 56 |
| Max Packet Length | 7 | 51 | 38 |
| Packet Length Mean | 8 | 56 | 40 |
| Average Packet Size | 10 | 59 | 34 |
| Bwd Packets/s | 37 | 60 | 6 |
| Packet Length Std | 2 | 61 | 23 |

## Lecture

Les trois classements ne disent pas la même chose, et les écarts-types de la permutation sont très petits : 0,00044 pour la première variable, dont la baisse de PR-AUC vaut 0,172. L'ordre du top 10 ne tient donc pas à une répétition particulière.

Destination Port est le point d'accord du vendredi. Le mélanger fait perdre 0,172 de PR-AUC, et il est premier au SHAP. La forêt le place seulement cinquième, derrière des mesures de taille de paquets.

Six variables sont dans le top 10 à la fois de la permutation et du SHAP : Destination Port, Bwd Packet Length Mean, Bwd Packet Length Std, Fwd IAT Min, Bwd Packet Length Max et Bwd Packet Length Min. Les trois premières baisses de PR-AUC sont 0,172, 0,130 et 0,098.

La forêt met en tête des tailles globales, Packet Length Variance puis Packet Length Std. Sur le vendredi, les mélanger ne fait presque rien, ou améliore légèrement la PR-AUC : la baisse vaut 0,0003 pour la variance, et −0,015 pour l'écart-type, qui termine au dernier rang. Ces colonnes sont très corrélées aux longueurs retour, qui restent, elles, en tête de la permutation. L'impureté compte plusieurs fois le même signal. La permutation, qui mélange une colonne pendant que les autres restent, ne le compte qu'une fois.

L'écart inverse existe aussi. Init_Win_bytes_backward et Init_Win_bytes_forward sont troisième et cinquième au SHAP, mais seulement 17e et 18e à la permutation, avec des baisses de 0,011 et 0,010. Bwd Packets/s est sixième au SHAP et 60e à la permutation : mélanger cette colonne fait monter la PR-AUC de 0,012. Une forte moyenne absolue SHAP, calculée sur 2 000 flux, ne signifie pas que la colonne porte le classement de tout le vendredi une fois les autres variables présentes.

Quatre variables entrent dans le top 10 de la permutation sans être dans celui du SHAP ni dans celui de la forêt : PSH Flag Count, Flow Duration, Fwd IAT Total et URG Flag Count. Leurs baisses vont de 0,030 à 0,025. Elles comptent pour la courbe précision-rappel du vendredi, et peu pour l'impureté apprise du lundi au mercredi.

## Limites

La forêt a été apprise du lundi au mercredi. Sa mesure d'impureté décrit cet apprentissage, pas le vendredi. La permutation et le SHAP décrivent le vendredi, mais pas les mêmes lignes : la permutation voit les 595 723 flux, le SHAP un échantillon de 2 000. Une baisse de PR-AUC dit qu'une variable sert au classement sur ce jour. Elle ne dit pas qu'il faudrait la retirer, et elle ne change pas le modèle officiel. Des variables corrélées, surtout les longueurs de paquets, se partagent l'importance : le rang de l'une peut monter quand l'autre est mélangée. Le seuil 0,01 n'entre pas dans la PR-AUC.

## Fichiers

- `outputs/metrics/test_permutation_importance.json`
- `outputs/tables/test_permutation_importance.csv`
- `outputs/tables/importance_methods_comparison.csv`
