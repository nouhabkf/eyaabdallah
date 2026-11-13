# Stabilité des explications SHAP

Deux expériences comparent le classement SHAP du XGBoost officiel. Aucune ne lit le vendredi. Aucune ne modifie le modèle officiel, son seuil ou ses hyperparamètres. Les fichiers officiels ne sont pas réécrits.

## Protocole

L'expérience A réentraîne trois fois sur tout le train officiel, du lundi au mercredi, avec les graines 42, 7 et 123. Seule cette graine change.

L'expérience B ne crée pas un nouveau découpage chronologique. Pour les graines 1001, 1002 et 1003, elle tire 70 % du train officiel, stratifié sur l'étiquette d'origine. L'imputation et `scale_pos_weight` sont ajustés sur ce sous-échantillon seul. La graine du modèle reste 42. Aucun flux du jeudi ni du vendredi n'entre dans ces apprentissages.

Les six classements expliquent le même échantillon de 2000 flux du jeudi, tiré avec la graine 42. La population de ce tirage s'arrête avant le groupe de lignes qui contient le vendredi. Les 25243 dernières lignes du jeudi, toutes normales, n'y sont pas. Les 2179 attaques du jeudi y sont toutes. L'explication est TreeExplainer, sur CPU, en moyenne des valeurs absolues du log-odds de l'attaque.

Le Jaccard compare les ensembles des dix premières variables. Spearman compare les rangs des 61 variables. Aucun seuil de réussite n'est appliqué.

## Expérience A, graines du modèle

| Run | Dix premières variables, dans l'ordre |
|---|---|
| graine-42 | Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward, Fwd IAT Min, Bwd Packet Length Min, Bwd Packets/s, Bwd Packet Length Max, min_seg_size_forward |
| graine-7 | Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward, Fwd IAT Min, Bwd Packet Length Min, Bwd Packets/s, Bwd Packet Length Max, min_seg_size_forward |
| graine-123 | Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward, Fwd IAT Min, Bwd Packet Length Min, Bwd Packets/s, Bwd Packet Length Max, min_seg_size_forward |

| Paire | Jaccard du top 10 | Spearman des 61 rangs |
|---|---:|---:|
| graine-42 / graine-7 | 1.000 | 1.000 |
| graine-42 / graine-123 | 1.000 | 1.000 |
| graine-7 / graine-123 | 1.000 | 1.000 |

Variables présentes dans les trois tops 10 : Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward, Fwd IAT Min, Bwd Packet Length Min, Bwd Packets/s, Bwd Packet Length Max, min_seg_size_forward.

## Expérience B, sensibilité au sous-échantillon d'entraînement

| Run | Dix premières variables, dans l'ordre |
|---|---|
| sous-echantillon-1001 | Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward, Bwd Packet Length Min, Bwd Packets/s, Bwd Packet Length Max, Fwd IAT Min, min_seg_size_forward |
| sous-echantillon-1002 | Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward, Bwd Packets/s, min_seg_size_forward, Bwd Packet Length Max, Fwd IAT Min, Fwd Header Length |
| sous-echantillon-1003 | Destination Port, Bwd Packet Length Std, Bwd Packet Length Mean, Init_Win_bytes_backward, Init_Win_bytes_forward, Bwd Packet Length Max, Bwd Packets/s, min_seg_size_forward, Bwd Packet Length Min, Fwd Header Length |

| Paire | Jaccard du top 10 | Spearman des 61 rangs |
|---|---:|---:|
| sous-echantillon-1001 / sous-echantillon-1002 | 0.818 | 0.911 |
| sous-echantillon-1001 / sous-echantillon-1003 | 0.818 | 0.903 |
| sous-echantillon-1002 / sous-echantillon-1003 | 0.818 | 0.898 |

Variables présentes dans les trois tops 10 : Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward, Bwd Packets/s, Bwd Packet Length Max, min_seg_size_forward.

## Lecture

Les trois graines produisent le même classement, et les mêmes valeurs SHAP : l'écart maximal entre les moyennes absolues est nul. Ce n'est pas une coïncidence à interpréter comme trois modèles différents tombés d'accord. Les hyperparamètres officiels fixent `subsample` et `colsample_bytree` à 1. Il ne reste alors aucun tirage de lignes ou de colonnes pour la graine. Avec ces réglages, la graine ne change pas l'explication.

Le choix des flux d'entraînement, lui, déplace la frontière du top 10. Le Jaccard moyen vaut 0,818 et le Spearman moyen des 61 rangs vaut 0,904. Le cœur reste fixe : Destination Port est toujours premier, Bwd Packet Length Std toujours deuxième, et les deux fenêtres initiales ainsi que Bwd Packet Length Mean restent dans les cinq premiers. Les variables instables sont celles du bas du classement. Fwd IAT Min passe du rang 6 sur le train complet aux rangs 9, 9 et 12. Fwd Header Length entre au rang 10 dans deux sous-échantillons et reste 11e dans le troisième. Bwd Packet Length Min sort du top 10 pour un des trois tirages, au rang 11.

## Limites

Le jeudi ne contient que 2 179 attaques, et l'échantillon SHAP en reprend une poignée. Un rang en bordure de top 10 peut donc bouger sans que le sens du modèle change. L'expérience A ne fait pas varier un aléa d'entraînement, parce que les hyperparamètres officiels n'en laissent pas. L'expérience B ne fait varier que le choix des flux du lundi au mercredi. Elle ne mesure pas un autre découpage des journées, et elle ne relance pas les partitions 70/15/15 qui auraient mélangé le vendredi. Les trois variables retirées dans l'ablation ne sont pas retestées ici. Le modèle de graine 42 est un nouvel ajustement : il n'écrase pas le fichier officiel. Les 25 243 dernières lignes normales du jeudi sont dans le groupe parquet qui contient aussi le vendredi ; elles ont été laissées de côté pour ne pas lire ce groupe.

## Fichiers

- `outputs/metrics/stability_explanations.json`
- `outputs/tables/stability_top10.csv`
- `outputs/tables/stability_rank_agreement.csv`
- `outputs/figures/stability_rank_heatmap.png`
- `outputs/models/stability/`
