# Préparation des données

Préparation du 13 novembre 2026. Les huit CSV de `C:\data\cic-ids2017\raw` ont été lus en binaire et leurs empreintes SHA-256 sont inchangées. Aucune ligne n'y a été écrite. Le fichier dérivé est `C:\data\cic-ids2017\interim\flows.parquet`. Les indices sont sous `C:\data\cic-ids2017\splits`. Le manifeste est `outputs/metrics/preparation_manifest.json`. Le journal des étapes est `docs/cleaning_log.md`.

Aucun modèle n'a été ajusté. Aucun SHAP n'a été calculé. Aucun modèle n'a été choisi. Le test n'a servi à aucune décision : ses effectifs sont seulement décrits.

Les trois algorithmes restent la régression logistique, la forêt aléatoire et XGBoost. Leurs pipelines sont construits et non ajustés.

## Opérations

| Étape | Résultat |
| --- | --- |
| Lignes brutes | 2 830 743 |
| Noms de colonnes | espaces de début et de fin retirés dans le fichier dérivé |
| `Fwd Header Length` | les deux colonnes sont égales sur les 2 830 743 lignes ; la seconde est absente du dérivé. Il reste 77 caractéristiques |
| Infinis | 2 773 cellules `Infinity` devenues manquantes. Le parquet en contient 0 |
| Manquants d'origine | 351 jetons `NaN` conservés comme manquants. Aucune ligne n'est supprimée pour ce motif |
| Étiquettes | texte d'origine conservé, y compris U+FFFD. Cible binaire : `BENIGN` = 0, toute autre étiquette = 1 |
| Profils contradictoires | 698 groupes, 7 020 lignes, exclus en entier |
| Première occurrence exacte | 302 757 copies ultérieures retirées. 2 520 966 lignes conservées |

Après nettoyage, les seules valeurs manquantes sont 1 562 cellules dans `Flow Bytes/s` et 1 562 cellules dans `Flow Packets/s`, soit 3 124. Ce total est la somme des 2 773 infinis remplacés et des 351 jetons manquants. Aucune imputation n'a été appliquée.

La comparaison des profils contradictoires et la déduplication portent sur les octets bruts, avant le stockage numérique. 28 011 contenus exacts non contradictoires traversaient deux ensembles ; seule leur première occurrence est conservée. 13 autres contenus exacts traversaient aussi deux ensembles, mais ils appartenaient déjà à un profil contradictoire et ont été exclus avec lui. 28 011 + 13 = 28 024, le total mesuré avant ces deux règles. Aucun contenu d'attaque exact ne traverse deux fichiers. Le fichier dérivé ne contient plus aucun de ces 698 profils.

Les valeurs sont stockées en float32, comme le protocole de calcul le fixe. 25 203 389 cellules ne sont pas représentables exactement dans ce format : ce sont surtout des entiers au-delà de 2^24 et des décimaux. Cette conversion ne change pas les lignes conservées. Elle change la valeur numérique que les modèles verront. Elle est inscrite au manifeste.

## Découpage chronologique officiel

L'affectation suit uniquement le fichier d'origine. Il n'y a pas de graine. Les indices `train.npy`, `validation.npy` et `test.npy` forment une partition des 2 520 966 lignes : 2 520 966 indices distincts.

| Ensemble | Lignes | Normal | Attack | Part d'attaque |
| --- | ---: | ---: | ---: | ---: |
| Apprentissage, lundi–mercredi | 1 527 133 | 1 324 353 | 202 780 | 13,28 % |
| Validation, jeudi | 398 110 | 395 931 | 2 179 | 0,55 % |
| Test, vendredi | 595 723 | 375 502 | 220 221 | 36,97 % |

| Famille | Apprentissage | Validation | Test |
| --- | ---: | ---: | ---: |
| FTP-Patator | 5 933 | 0 | 0 |
| SSH-Patator | 3 219 | 0 | 0 |
| DoS slowloris | 5 384 | 0 | 0 |
| DoS Slowhttptest | 5 228 | 0 | 0 |
| DoS Hulk | 172 719 | 0 | 0 |
| DoS GoldenEye | 10 286 | 0 | 0 |
| Heartbleed | 11 | 0 | 0 |
| Web Attack U+FFFD Brute Force | 0 | 1 470 | 0 |
| Web Attack U+FFFD XSS | 0 | 652 | 0 |
| Web Attack U+FFFD Sql Injection | 0 | 21 | 0 |
| Infiltration | 0 | 36 | 0 |
| Bot | 0 | 0 | 1 953 |
| PortScan | 0 | 0 | 90 255 |
| DDoS | 0 | 0 | 128 013 |

Aucune famille n'a disparu. Celles du test sont absentes de l'apprentissage, et celles de la validation aussi.

## Découpage groupé complémentaire

Graine 42, générateur `random.Random` de Python 3.12. Pour chaque étiquette d'origine, dans l'ordre Unicode, les positions chronologiques sont mélangées. La validation et le test reçoivent chacun `n * 15 // 100` lignes, l'apprentissage le reste. Un contenu exact ne peut pas être dans deux ensembles : chaque ligne conservée est déjà un contenu unique, et les indices ne se recouvrent pas.

Ce découpage ne choisit ni le modèle officiel, ni le seuil officiel. Ses effectifs de validation et de test sont égaux famille par famille à cause de la règle entière. Les indices, eux, diffèrent : leurs empreintes SHA-256 ne sont pas les mêmes.

| Ensemble | Lignes | Normal | Attack | Part d'attaque |
| --- | ---: | ---: | ---: | ---: |
| Apprentissage | 1 764 696 | 1 467 052 | 297 644 | 16,87 % |
| Validation | 378 135 | 314 367 | 63 768 | 16,86 % |
| Test | 378 135 | 314 367 | 63 768 | 16,86 % |

| Famille | Apprentissage | Validation | Test |
| --- | ---: | ---: | ---: |
| DoS Hulk | 120 905 | 25 907 | 25 907 |
| DDoS | 89 611 | 19 201 | 19 201 |
| PortScan | 63 179 | 13 538 | 13 538 |
| DoS GoldenEye | 7 202 | 1 542 | 1 542 |
| FTP-Patator | 4 155 | 889 | 889 |
| DoS slowloris | 3 770 | 807 | 807 |
| DoS Slowhttptest | 3 660 | 784 | 784 |
| SSH-Patator | 2 255 | 482 | 482 |
| Bot | 1 369 | 292 | 292 |
| Web Attack U+FFFD Brute Force | 1 030 | 220 | 220 |
| Web Attack U+FFFD XSS | 458 | 97 | 97 |
| Infiltration | 26 | 5 | 5 |
| Web Attack U+FFFD Sql Injection | 15 | 3 | 3 |
| Heartbleed | 9 | 1 | 1 |

Les quatorze familles sont dans les trois ensembles.

## Pipelines, non ajustés

| Modèle | Transformations | Réglage de classe |
| --- | --- | --- |
| Régression logistique | médiane, puis standardisation | `class_weight="balanced"` |
| Forêt aléatoire | médiane, sans standardisation | `class_weight="balanced"` |
| XGBoost | médiane, sans standardisation, `device="cpu"`, `tree_method="hist"` | `scale_pos_weight` non inscrit : il sera `n_normal / n_attack` du train au moment de l'ajustement |

`fit` n'a pas été appelé sur ces pipelines. Un test automatique ajuste seulement l'imputation et la standardisation sur quatre lignes synthétiques, et vérifie que la moyenne apprise ne contient pas une cinquième ligne tenue à l'écart. Les données CIC-IDS2017 ne servent pas à cet essai.

## Tests

Les tests couvrent la configuration, le blocage de l'entraînement, les règles de conservation, les pipelines non ajustés, les effectifs chronologiques, l'absence d'infini et de profil contradictoire dans le dérivé, et la partition des deux jeux d'indices.

L'entraînement n'est pas commencé. Il attend votre validation.
