# Entraînement et sélection sur la validation chronologique

Les trois modèles ont été ajustés. Le test n'a pas été lu pour un score, un hyperparamètre ou le choix du modèle. Le seuil officiel `1 × FPR + 2 × FNR` n'a pas été choisi : les métriques binaires ci-dessous sont au seuil 0,5 demandé pour cette comparaison. SHAP et l'audit des variables n'ont pas été calculés. Le parquet préparé et les indices n'ont pas été modifiés.

Modèle retenu : **xgboost**, par PR-AUC de validation. Règle de départage, dans l'ordre : rappel au seuil 0,5 le plus haut, puis temps de réentraînement le plus court, puis l'ordre des familles dans la configuration. Cette dernière clé ne sert que si les trois critères précédents sont égaux.

## Données

Train chronologique lundi–mercredi : 1 527 133 lignes, dont 202 780 attaques. Validation jeudi : 398 110 lignes, dont 2 179 attaques (0.5473% de la validation). Les familles mesurées coïncident avec le manifeste de préparation.

Caractéristiques lues : 77. Conservées après filtres appris sur le train seul : 61. Colonnes constantes écartées : Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, CWE Flag Count, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate. Colonnes doublons exactes écartées : SYN Flag Count, Avg Fwd Segment Size, Avg Bwd Segment Size, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets.

Sous-échantillon de recherche : 200 000 lignes du train, graine 42, dont 26 557 attaques. Trois plis stratifiés internes, graine 42. La validation officielle n'entre dans aucun pli.

## Hyperparamètres retenus

Le score de recherche est la moyenne d'average precision des trois plis, calculée avec `predict_proba` de la classe attaque. Ce score choisit les hyperparamètres. Il ne choisit pas le modèle. Le modèle est choisi plus bas, sur la validation.

- **logistic_regression** : C = 10.0. PR-AUC interne = 0.992043. Recherche : 33.1 s. Réentraînement sur tout le train : 70.8 s.
  Itérations lbfgs : 439.
- **random_forest** : n_estimators = 200, min_samples_leaf = 1, max_features = sqrt, max_depth = 20. PR-AUC interne = 0.999959. Recherche : 580.0 s. Réentraînement sur tout le train : 229.7 s.
- **xgboost** : subsample = 1.0, n_estimators = 100, min_child_weight = 1, max_depth = 8, learning_rate = 0.2, colsample_bytree = 1.0. PR-AUC interne = 0.999990. Recherche : 96.5 s. Réentraînement sur tout le train : 20.9 s.
  `scale_pos_weight` calculé sur le train complet au moment du fit : 6.530984 (1 324 353 / 202 780).

Ces PR-AUC internes sont calculées sur des plis du lundi–mercredi, donc sur les mêmes familles d'attaques que l'apprentissage. Elles saturent pour les trois modèles et ne servent pas à les départager. La validation du jeudi ne contient que des familles absentes du train. C'est cette PR-AUC de transfert qui choisit le modèle. L'écart entre le score interne et le score du jeudi, très marqué pour la régression logistique, décrit ce changement de familles. L'ajustement lui-même s'est terminé sans avertissement.

## Comparaison sur la validation, seuil 0,5

| Modèle | Retenu | PR-AUC | Précision | Rappel | F1 | TN | FP | FN | TP | Entraînement (s) | Inférence moyenne (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression |  | 0.034144 | 0.003050 | 0.011932 | 0.004858 | 387 432 | 8 499 | 2 153 | 26 | 70.8 | 0.299 |
| random_forest |  | 0.790506 | 0.911258 | 0.631482 | 0.746002 | 395 797 | 134 | 803 | 1 376 | 229.7 | 1.510 |
| xgboost | oui | 0.932439 | 0.953207 | 0.832033 | 0.888508 | 395 842 | 89 | 366 | 1 813 | 20.9 | 0.359 |

La colonne Retenu indique le modèle choisi par la PR-AUC. Les trois scores sont séparés, donc la règle de départage n'a pas eu à trancher. Le seuil 0,5 n'entre pas dans ce choix : il décrit un seul point de fonctionnement. Le seuil officiel, qui minimise `1 × FPR + 2 × FNR`, n'est pas encore fixé et pourra déplacer les comptes de la matrice. Il ne changera pas la PR-AUC.

L'inférence est la moyenne de cinq passages `predict_proba` sur toute la validation. Le test n'a pas été chronométré. Les courbes précision-rappel sont dans `outputs/figures/validation_precision_recall.png`. La ligne de prévalence rappelle le taux d'attaque du jeudi ; ce n'est pas un quatrième modèle.

## Ressources

Machine : Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 20 processeurs logiques. Mémoire physique : 23.6 Gio, disponible au départ : 4.2 Gio, à la fin : 7.0 Gio. Durée totale : 17.5 min. Graines : numpy et recherche aléatoire 42, sous-échantillon 42, plis 42, modèles 42.

Versions : numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, pyarrow 26.0.1, joblib 1.6.0, matplotlib 3.11.2.

## Problèmes techniques

Aucun avertissement de convergence ni échec d'ajustement n'a été enregistré.

Les tailles des modèles enregistrés : logistic_regression 2 860 octets, random_forest 6.2 Mio, xgboost 141 665 octets.

## Ce qui reste fermé

Le seuil de décision officiel, l'évaluation du vendredi, SHAP et l'audit des variables attendent une autorisation. Les trois modèles sont conservés.
