# Vérification de la conversion float32

Mesure en lecture seule sur les 2 520 966 lignes conservées, avant toute autorisation d'entraînement. Aucun modèle n'a été ajusté. Aucun SHAP n'a été calculé. Les trois modèles restent Logistic Regression, Random Forest et XGBoost.

Les CSV bruts, le parquet, les indices, la configuration et les pipelines n'ont pas été modifiés. Les chiffres exacts sont dans `outputs/metrics/float32_verification.json`.

## Définition mesurée

Pour chaque cellule finie des 77 caractéristiques, la valeur d'origine est le nombre lu en float64 depuis le jeton brut, et la valeur stockée est `float(float32(origine))`. Une cellule est modifiée lorsque ces deux nombres diffèrent. Les cellules manquantes et infinies ne sont pas comptées ici : elles sont déjà devenues manquantes, 3 124 cellules, uniquement dans Flow Bytes/s et Flow Packets/s.

L'erreur absolue moyenne est calculée sur toutes les valeurs finies, y compris celles qui restent identiques. L'erreur relative moyenne est la moyenne de |erreur| / |valeur| sur les valeurs finies non nulles. L'écart-type utilisé pour l'erreur en unités standard est celui des valeurs d'origine, calculé en float64.

Le passage a retrouvé 2 520 966 lignes et zéro divergence entre les deux colonnes Fwd Header Length.

## 1. Valeurs réellement modifiées

**25 203 389 cellules sont modifiées.**

C'est exactement le nombre indiqué dans le rapport de préparation. Sur ces fichiers, « non représentable exactement en float32 » et « valeur réellement changée par la conversion » désignent le même ensemble : la valeur stockée est le float32 le plus proche, et elle diffère de l'original.

| Quantité | Valeur |
|---|---:|
| Lignes conservées | 2 520 966 |
| Cellules finies | 194 111 258 |
| Cellules modifiées | 25 203 389 |
| Part des cellules finies | 12,98 % |
| Caractéristiques inchangées | 35 sur 77 |
| Erreur absolue maximale, toutes colonnes | 392 |
| Erreur absolue moyenne, toutes cellules finies | 0,00765 |
| Erreur relative maximale | 5,960 × 10⁻⁸ |

L'erreur relative maximale est celle de l'arrondi float32 usuel (2⁻²⁴ ≈ 5,96 × 10⁻⁸). Aucune cellule n'est déplacée au-delà du float32 le plus proche.

## 2. Erreur par caractéristique

L'erreur absolue moyenne inclut les valeurs inchangées. L'erreur relative moyenne exclut les zéros exacts. La colonne « entiers décalés » compte les jetons entiers dont la valeur stockée s'éloigne d'au moins 1.

| Caractéristique | Valeurs modifiées | Erreur absolue max | Erreur absolue moyenne | Erreur relative max | Erreur relative moyenne | Entiers décalés d’au moins 1 |
|---|---:|---:|---:|---:|---:|---:|
| Destination Port | 0 | 0 | 0 | 0 | 0 | 0 |
| Flow Duration | 376 026 | 4.000e+0 | 0.309266 | 5.960e-8 | 3.605e-9 | 376 026 |
| Total Fwd Packets | 0 | 0 | 0 | 0 | 0 | 0 |
| Total Backward Packets | 0 | 0 | 0 | 0 | 0 | 0 |
| Total Length of Fwd Packets | 0 | 0 | 0 | 0 | 0 | 0 |
| Total Length of Bwd Packets | 16 | 3.100e+1 | 7.656e-5 | 5.385e-8 | 2.610e-13 | 16 |
| Fwd Packet Length Max | 0 | 0 | 0 | 0 | 0 | 0 |
| Fwd Packet Length Min | 0 | 0 | 0 | 0 | 0 | 0 |
| Fwd Packet Length Mean | 593 568 | 0.0002365 | 7.241e-7 | 5.881e-8 | 6.998e-9 | 0 |
| Fwd Packet Length Std | 949 426 | 0.000243188 | 1.678e-6 | 5.946e-8 | 2.137e-8 | 0 |
| Bwd Packet Length Max | 0 | 0 | 0 | 0 | 0 | 0 |
| Bwd Packet Length Min | 0 | 0 | 0 | 0 | 0 | 0 |
| Bwd Packet Length Mean | 440 198 | 0.000121797 | 3.339e-6 | 5.894e-8 | 5.386e-9 | 0 |
| Bwd Packet Length Std | 779 973 | 0.00033075 | 8.391e-6 | 5.949e-8 | 2.204e-8 | 0 |
| Flow Bytes/s | 2 115 966 | 6.400e+1 | 0.00780223 | 5.952e-8 | 2.059e-8 | 126 |
| Flow Packets/s | 2 415 217 | 5.700e-2 | 6.324e-4 | 5.956e-8 | 2.128e-8 | 0 |
| Flow IAT Mean | 1 188 735 | 4.000e+0 | 0.0199662 | 5.960e-8 | 1.196e-8 | 2 767 |
| Flow IAT Std | 1 498 960 | 3.962e+0 | 0.0177833 | 5.949e-8 | 1.873e-8 | 0 |
| Flow IAT Max | 20 607 | 4.000e+0 | 0.0125857 | 5.960e-8 | 2.201e-10 | 20 607 |
| Flow IAT Min | 1 455 | 4.000e+0 | 7.430e-4 | 5.935e-8 | 1.412e-11 | 1 455 |
| Fwd IAT Total | 48 275 | 4.000e+0 | 0.0368589 | 5.959e-8 | 6.025e-10 | 48 275 |
| Fwd IAT Mean | 467 137 | 4.000e+0 | 0.0164624 | 5.960e-8 | 6.151e-9 | 8 574 |
| Fwd IAT Std | 824 655 | 3.979e+0 | 0.0152077 | 5.952e-8 | 1.732e-8 | 0 |
| Fwd IAT Max | 20 271 | 4.000e+0 | 0.012383 | 5.960e-8 | 2.862e-10 | 20 271 |
| Fwd IAT Min | 8 530 | 4.000e+0 | 0.00568393 | 5.957e-8 | 1.236e-10 | 8 530 |
| Bwd IAT Total | 44 517 | 4.000e+0 | 0.0347383 | 5.960e-8 | 7.037e-10 | 44 517 |
| Bwd IAT Mean | 492 073 | 4.000e+0 | 0.0155227 | 5.960e-8 | 8.224e-9 | 7 478 |
| Bwd IAT Std | 719 504 | 3.997e+0 | 0.0115622 | 5.958e-8 | 1.934e-8 | 0 |
| Bwd IAT Max | 16 523 | 4.000e+0 | 0.0102687 | 5.960e-8 | 2.913e-10 | 16 523 |
| Bwd IAT Min | 7 374 | 4.000e+0 | 0.00511828 | 5.959e-8 | 1.351e-10 | 7 374 |
| Fwd PSH Flags | 0 | 0 | 0 | 0 | 0 | 0 |
| Bwd PSH Flags | 0 | 0 | 0 | 0 | 0 | 0 |
| Fwd URG Flags | 0 | 0 | 0 | 0 | 0 | 0 |
| Bwd URG Flags | 0 | 0 | 0 | 0 | 0 | 0 |
| Fwd Header Length | 32 | 3.920e+2 | 3.392e-4 | 3.576e-8 | 2.681e-13 | 32 |
| Bwd Header Length | 20 | 3.000e+1 | 4.006e-5 | 3.576e-8 | 2.278e-13 | 20 |
| Fwd Packets/s | 2 417 217 | 5.700e-2 | 5.307e-4 | 5.956e-8 | 2.120e-8 | 0 |
| Bwd Packets/s | 2 162 413 | 4.200e-2 | 1.060e-4 | 5.953e-8 | 2.142e-8 | 0 |
| Min Packet Length | 0 | 0 | 0 | 0 | 0 | 0 |
| Max Packet Length | 0 | 0 | 0 | 0 | 0 | 0 |
| Packet Length Mean | 1 528 413 | 0.000118844 | 3.492e-6 | 5.911e-8 | 1.756e-8 | 0 |
| Packet Length Std | 1 936 655 | 0.000242687 | 7.004e-6 | 5.958e-8 | 2.269e-8 | 0 |
| Packet Length Variance | 1 683 684 | 5.507e-1 | 0.00980192 | 5.945e-8 | 2.183e-8 | 0 |
| FIN Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| SYN Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| RST Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| PSH Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| ACK Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| URG Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| CWE Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| ECE Flag Count | 0 | 0 | 0 | 0 | 0 | 0 |
| Down/Up Ratio | 0 | 0 | 0 | 0 | 0 | 0 |
| Average Packet Size | 754 492 | 0.000118391 | 2.897e-6 | 5.896e-8 | 7.969e-9 | 0 |
| Avg Fwd Segment Size | 593 568 | 0.0002365 | 7.241e-7 | 5.881e-8 | 6.998e-9 | 0 |
| Avg Bwd Segment Size | 440 198 | 0.000121797 | 3.339e-6 | 5.894e-8 | 5.386e-9 | 0 |
| Fwd Avg Bytes/Bulk | 0 | 0 | 0 | 0 | 0 | 0 |
| Fwd Avg Packets/Bulk | 0 | 0 | 0 | 0 | 0 | 0 |
| Fwd Avg Bulk Rate | 0 | 0 | 0 | 0 | 0 | 0 |
| Bwd Avg Bytes/Bulk | 0 | 0 | 0 | 0 | 0 | 0 |
| Bwd Avg Packets/Bulk | 0 | 0 | 0 | 0 | 0 | 0 |
| Bwd Avg Bulk Rate | 0 | 0 | 0 | 0 | 0 | 0 |
| Subflow Fwd Packets | 0 | 0 | 0 | 0 | 0 | 0 |
| Subflow Fwd Bytes | 0 | 0 | 0 | 0 | 0 | 0 |
| Subflow Bwd Packets | 0 | 0 | 0 | 0 | 0 | 0 |
| Subflow Bwd Bytes | 102 | 3.100e+1 | 2.955e-4 | 5.835e-8 | 1.579e-12 | 102 |
| Init_Win_bytes_forward | 0 | 0 | 0 | 0 | 0 | 0 |
| Init_Win_bytes_backward | 0 | 0 | 0 | 0 | 0 | 0 |
| act_data_pkt_fwd | 0 | 0 | 0 | 0 | 0 | 0 |
| min_seg_size_forward | 34 | 5.000e+0 | 2.737e-5 | 1.192e-8 | 1.428e-13 | 34 |
| Active Mean | 116 184 | 2.000e+0 | 3.597e-4 | 5.958e-8 | 5.281e-9 | 20 |
| Active Std | 204 825 | 1.667e+0 | 9.554e-4 | 5.950e-8 | 2.147e-8 | 0 |
| Active Max | 162 | 4.000e+0 | 7.140e-5 | 5.956e-8 | 1.211e-11 | 162 |
| Active Min | 17 | 2.000e+0 | 7.537e-6 | 5.731e-8 | 1.221e-12 | 17 |
| Idle Mean | 104 365 | 4.000e+0 | 0.0204645 | 5.960e-8 | 5.248e-9 | 13 912 |
| Idle Std | 197 708 | 3.840e+0 | 0.0015731 | 5.946e-8 | 1.865e-8 | 0 |
| Idle Max | 18 375 | 4.000e+0 | 0.0114254 | 5.960e-8 | 8.735e-10 | 18 375 |
| Idle Min | 15 919 | 4.000e+0 | 0.00998189 | 5.960e-8 | 7.600e-10 | 15 919 |

## 3. Caractéristiques les plus affectées

Trois classements, parce qu'ils ne désignent pas les mêmes colonnes.

**Par erreur absolue maximale.** Ces écarts portent sur quelques cellules de très grande magnitude, souvent des artefacts déjà connus de CIC-FlowMeter (longueurs d'en-tête négatives ou de plusieurs milliards).

| Caractéristique | Cellules | Erreur absolue max | Valeur d'origine la plus écartée | Valeur stockée | Erreur relative max |
|---|---:|---:|---:|---:|---:|
| Fwd Header Length | 32 | 392 | −32 212 234 632 | −32 212 234 240 | 3,58 × 10⁻⁸ |
| Flow Bytes/s | 2 115 966 | 64 | 2 071 000 000 | 2 071 000 064 | 5,95 × 10⁻⁸ |
| Total Length of Bwd Packets | 16 | 31 | 575 681 633 | 575 681 664 | 5,38 × 10⁻⁸ |
| Subflow Bwd Bytes | 102 | 31 | 575 681 633 | 575 681 664 | 5,84 × 10⁻⁸ |
| Bwd Header Length | 20 | 30 | −1 073 741 218 | −1 073 741 248 | 3,58 × 10⁻⁸ |
| min_seg_size_forward | 34 | 5 | −536 870 661 | −536 870 656 | 1,19 × 10⁻⁸ |

**Par nombre de cellules modifiées.** Ce sont des débits et des statistiques continues. Leur erreur absolue maximale reste inférieure à 0,06, sauf Flow Bytes/s et Packet Length Variance (0,55).

| Caractéristique | Cellules modifiées | Erreur absolue max |
|---|---:|---:|
| Fwd Packets/s | 2 417 217 | 0,057 |
| Flow Packets/s | 2 415 217 | 0,057 |
| Bwd Packets/s | 2 162 413 | 0,042 |
| Flow Bytes/s | 2 115 966 | 64 |
| Packet Length Std | 1 936 655 | 0,000243 |
| Packet Length Variance | 1 683 684 | 0,551 |

**Par perturbation après standardisation.** L'erreur absolue maximale divisée par l'écart-type de la colonne d'origine. C'est l'échelle vue par Logistic Regression après StandardScaler.

| Caractéristique | Erreur max en écarts-types |
|---|---:|
| Bwd Header Length | 1,95 × 10⁻⁵ |
| Fwd Header Length | 1,76 × 10⁻⁵ |
| Subflow Bwd Bytes | 1,29 × 10⁻⁵ |
| Total Length of Bwd Packets | 1,29 × 10⁻⁵ |
| min_seg_size_forward | 4,35 × 10⁻⁶ |

Pour les durées et les intervalles entre paquets, l'erreur absolue maximale est 4 et l'erreur en écarts-types reste de l'ordre de 10⁻⁷.

## 4. Entiers, durées, paquets, ports

Cinquante-trois caractéristiques sont entièrement entières dans les jetons bruts. Trente-cinq le restent exactement en float32, dont tout ce qui porte une information de petit entier :

- Destination Port : 0 cellule modifiée. Les ports tiennent dans 0–65 535, bien en dessous de 2²⁴ = 16 777 216, limite des entiers exacts en float32.
- Totaux de paquets, longueurs de paquets min/max, drapeaux, fenêtres initiales, Down/Up Ratio, volumes avant et sous-flux avant : 0 cellule modifiée.
- Total Length of Fwd Packets et Subflow Fwd Bytes : exacts.

Dix-huit colonnes entièrement entières ont au moins une valeur modifiée. En y ajoutant six colonnes mixtes qui contiennent aussi des entiers arrondis, **611 132 jetons entiers sont décalés d'au moins 1**. Ce décalage ne retire pas une précision utile pour la détection.

Les durées et les temps entre paquets sont en microsecondes. Flow Duration passe au pire de 119 719 148 à 119 719 152 : 4 µs sur environ 120 s. L'erreur relative est 5,96 × 10⁻⁸. 376 026 durées perdent leur identité entière, toujours d'au plus 4 µs. Le même plafond de 4 vaut pour les totaux et maxima d'IAT et d'Idle. Active Mean et Active Min sont décalés d'au plus 2. Un écart de quelques microsecondes ne sépare pas un flux normal d'un flux d'attaque dans ce jeu : les familles se distinguent à l'échelle de la milliseconde ou de la seconde.

Les écarts absolus de 5 à 392 portent sur des valeurs qui ne sont déjà pas des mesures physiques crédibles : en-tête de −32 milliards, taille de segment de −537 millions, total d'octets arrière de 576 millions. L'erreur relative y reste inférieure à 6 × 10⁻⁸. Les valeurs ordinaires de ces colonnes, celles de quelques dizaines ou milliers d'octets, restent exactes. Seuls 16 à 102 jetons, selon la colonne, quittent l'entier d'origine.

## 5. Effet possible sur les trois modèles

Aucun des trois modèles n'a été entraîné. La conclusion vient de la mesure et du comportement des bibliothèques installées (scikit-learn 1.9.1, XGBoost 3.4.1).

**Random Forest.** `BaseDecisionTree.fit` force `dtype=float32` dans `check_array`. La forêt reçoit donc les mêmes valeurs que celles déjà stockées, que le disque contienne du float32 ou du float64. Stocker du float64 ne change pas ses coupures.

**XGBoost.** La bibliothèque convertit les entrées en float32 avant l'histogramme. Avec `tree_method=hist` et 256 bins, la largeur d'un bin est (max − min) / 256. Le plus grand rapport erreur absolue maximale / largeur de bin est 1,31 × 10⁻⁵, sur Packet Length Std (erreur 0,000243 pour un bin de 18,5). Pour Flow Duration, l'erreur de 4 µs vaut 8,5 × 10⁻⁶ d'un bin de 468 750 µs. Une valeur ne changerait de bin que si elle se trouvait déjà à cette distance d'une frontière. Comme XGBoost reconvertit de toute façon en float32, le dtype de stockage ne change pas les bins qu'il voit.

**Logistic Regression.** C'est le seul des trois qui peut rester en float64 si on lui fournit du float64 : le solveur accepte les deux types et conserve celui de la matrice. Après standardisation, la plus grande perturbation mesurée est 1,95 × 10⁻⁵ écart-type, sur Bwd Header Length. Sur une combinaison linéaire de 77 variables standardisées, un déplacement de cet ordre ne déplace la probabilité que pour un score déjà collé au seuil, bien en deçà de ce qui changerait la PR-AUC ou le critère de seuil 1 × FPR + 2 × FNR. Ce n'est pas un delta mesuré sur un modèle ajusté : c'est une borne sur les entrées. L'entraînement n'a pas été lancé pour la vérifier empiriquement.

Conséquence : passer les colonnes sensibles en float64 ne changerait ni Random Forest ni XGBoost, et ne donnerait à Logistic Regression qu'une correction inférieure à 2 × 10⁻⁵ écart-type sur la pire caractéristique.

## 6. Mémoire et temps

La matrice des 77 caractéristiques, 2 520 966 lignes :

| | float32 | float64 |
|---|---:|---:|
| Octets | 776 457 528 | 1 552 915 056 |
| Taille | 741 Mio | 1 481 Mio |

La matrice d'entraînement seule, 1 527 133 lignes, occupe 449 Mio en float32 et 897 Mio en float64. Logistic Regression garde en pratique deux matrices, l'imputée et la standardisée : environ 0,9 Gio contre 1,8 Gio pour l'entraînement seul. Random Forest et XGBoost recopient ensuite en float32, donc le supplément float64 est payé au chargement puis abandonné.

Le seul temps mesuré est `StandardScaler.fit` sur 400 000 × 77 tirages normaux, graine 42, pas sur CIC-IDS2017 et pas sur un modèle :

| | float32 | float64 |
|---|---:|---:|
| Secondes | 0,275 | 0,325 |

Le float64 est 18 % plus lent sur ce calcul. Rapporté linéairement à la taille de l'entraînement, le scaler seul passerait d'environ 1,05 s à 1,24 s. Ce n'est pas un temps d'entraînement. Le temps des arbres est dominé par la recherche de coupures après conversion en float32 ; le dtype de stockage ne le change pas. Aucun des trois modèles n'a été chronométré.

## Décision proposée

**Conserver float32 pour les 77 caractéristiques.**

Le float64 par colonne n'apporte pas de précision aux deux modèles à arbres, qui reconvertissent en float32. Il n'apporte à Logistic Regression qu'une correction inférieure à 2 × 10⁻⁵ écart-type. Les ports, drapeaux, compteurs de paquets et fenêtres restent des entiers exacts. Les entiers qui bougent sont des durées déjà au-delà de 2²⁴, d'au plus 4 µs, ou des artefacts de très grande magnitude dont l'erreur relative reste celle de l'arrondi float32. Doubler la matrice, de 741 Mio à 1 481 Mio, ne change pas les résultats attendus.

Cette décision n'est pas appliquée. La configuration indique déjà float32 ; le parquet n'est pas réécrit.

L'entraînement n'est pas lancé. Il attend une autorisation explicite, après acceptation de cette décision.
