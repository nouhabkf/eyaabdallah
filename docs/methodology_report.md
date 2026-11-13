# Rapport méthodologique

Analyse du 13 novembre 2026, en lecture seule. Aucune ligne n'a été supprimée. Les CSV bruts n'ont pas été modifiés. Aucun modèle n'a été ajusté. Aucun découpage n'a été enregistré.

Les trois algorithmes restent prévus : régression logistique, forêt aléatoire et XGBoost. Les trois peuvent être entraînés et évalués sur les volumes mesurés, dès lors que l'ensemble d'entraînement contient les deux classes.

Les chiffres viennent de `scripts/audit_methodology.py` et de `outputs/metrics/methodology_audit.json`.

La comparaison ultérieure des découpages chronologiques est dans `docs/protocol_comparison.md`. Elle ne modifie pas les chiffres de ce rapport et n'applique pas la recommandation ci-dessous.

## Recommandation, non appliquée

Le découpage principal recommandé est le suivant, après validation :

1. Un doublon exact est une ligne identique, étiquette comprise. En conserver un seul exemplaire dans le jeu dérivé. Il resterait 2 522 362 lignes.
2. Regrouper aussi les lignes qui ont les mêmes variables mais des étiquettes différentes. Ces groupes contradictoires, 698 contenus et 7 020 lignes, ne serviraient ni à l'entraînement ni au test officiel. Ils seraient seulement comptés dans le rapport.
3. Répartir les autres contenus distincts en 70 % / 15 % / 15 %, de façon stratifiée sur l'étiquette, sans qu'un même contenu puisse traverser deux ensembles.
4. Ajouter un contrôle par jour, publié à part. Il ne choisirait ni le modèle, ni le seuil, ni les variables.

Un découpage uniquement par jour ne permet pas d'évaluer les familles d'attaque apprises : chacune n'existe que dans un seul fichier. L'absence de colonne d'heure empêche un découpage plus fin à l'intérieur d'une journée.

## 1. Doublons exacts

Une copie exacte reprend toutes les colonnes, y compris ` Label`. Deux copies exactes ont donc la même étiquette. Le contrôle porte sur 2 830 743 lignes.

| Mesure | Valeur |
| --- | ---: |
| Lignes | 2 830 743 |
| Contenus exacts distincts | 2 522 362 |
| Copies supplémentaires | 308 381 |
| Lignes après conservation d'un seul exemplaire | 2 522 362 |

### Par classe et par famille

| Étiquette | Lignes actuelles | Après un exemplaire | Copies retirées |
| --- | ---: | ---: | ---: |
| BENIGN | 2 273 097 | 2 096 484 | 176 613 |
| DoS Hulk | 231 073 | 172 849 | 58 224 |
| PortScan | 158 930 | 90 819 | 68 111 |
| DDoS | 128 027 | 128 016 | 11 |
| DoS GoldenEye | 10 293 | 10 286 | 7 |
| FTP-Patator | 7 938 | 5 933 | 2 005 |
| SSH-Patator | 5 897 | 3 219 | 2 678 |
| DoS slowloris | 5 796 | 5 385 | 411 |
| DoS Slowhttptest | 5 499 | 5 228 | 271 |
| Bot | 1 966 | 1 953 | 13 |
| Web Attack, U+FFFD, Brute Force | 1 507 | 1 470 | 37 |
| Web Attack, U+FFFD, XSS | 652 | 652 | 0 |
| Infiltration | 36 | 36 | 0 |
| Web Attack, U+FFFD, Sql Injection | 21 | 21 | 0 |
| Heartbleed | 11 | 11 | 0 |

Avant : 2 273 097 Normal (80,34 %) et 557 646 Attack (19,66 %). Après un exemplaire de chaque ligne exacte : 2 096 484 Normal (83,12 %) et 425 878 Attack (16,88 %). La part d'attaques baisse surtout parce que PortScan et DoS Hulk contiennent beaucoup de copies. Aucune famille ne disparaît. Heartbleed, Sql Injection, Infiltration et XSS restent inchangés.

### À l'intérieur de chaque fichier

| Fichier | Copies exactes supplémentaires | Dont Normal | Dont Attack |
| --- | ---: | ---: | ---: |
| Lundi | 26 935 | 26 935 | 0 |
| Mardi | 24 065 | 19 382 | 4 683 |
| Mercredi | 81 909 | 22 996 | 58 913 |
| Jeudi matin | 6 066 | 6 029 | 37 |
| Jeudi après-midi | 35 630 | 35 630 | 0 |
| Vendredi matin | 6 888 | 6 875 | 13 |
| Vendredi PortScan | 72 353 | 4 242 | 68 111 |
| Vendredi DDoS | 2 633 | 2 622 | 11 |

Les copies d'attaque restent dans le fichier de leur famille. Les plus nombreuses sont PortScan (68 111) et DoS Hulk (58 224).

### Entre fichiers et entre jours

33 563 contenus exacts apparaissent dans plusieurs fichiers. Ils représentent 139 843 lignes, toutes `BENIGN`. Aucune ligne d'attaque exacte ne traverse deux jours.

Chaque paire de fichiers partage au moins un contenu bénin. Les intersections les plus grandes, en contenus distincts, sont mardi-mercredi (8 047), mercredi-jeudi après-midi (6 947) et mercredi-vendredi matin (6 639). La plus petite est lundi-mardi (53). Les 28 paires sont dans le JSON d'audit.

## 2. Mêmes variables, étiquettes différentes

Ce cas est distinct du doublon exact. Les variables, sans ` Label`, sont identiques, mais les étiquettes ne le sont pas.

698 contenus de variables portent deux étiquettes. Aucun n'en porte trois. Ces contenus couvrent 7 020 lignes.

| Paire d'étiquettes | Contenus | Lignes | Fichiers disjoints | Au moins un fichier commun |
| --- | ---: | ---: | ---: | ---: |
| BENIGN et PortScan | 564 | 1 618 | 564 | 0 |
| BENIGN et DoS Hulk | 130 | 5 388 | 83 | 47 |
| BENIGN et DDoS | 3 | 9 | 3 | 0 |
| BENIGN et DoS slowloris | 1 | 5 | 0 | 1 |

Le mercredi contient 117 lignes dont l'étiquette contredit la première étiquette déjà vue pour les mêmes variables dans ce fichier. Les autres fichiers n'ont aucun conflit interne de ce type.

Conserver un seul exemplaire des lignes exactes ne retire pas ces contradictions : la ligne bénigne et la ligne d'attaque diffèrent par l'étiquette, donc ce ne sont pas des doublons exacts. Un découpage qui les sépare montrerait au modèle le même profil avec deux classes.

## 3. Stratégies de découpage

Aucune n'est appliquée. Les effectifs de la stratégie A utilisent les lignes actuelles. Ceux de la stratégie B utilisent les 2 522 362 contenus exacts distincts.

### A. Par jour ou par fichier

Découpage chronologique étudié : entraînement lundi, mardi et mercredi ; validation jeudi ; test vendredi.

| Ensemble | Normal | Attack | Familles d'attaque présentes |
| --- | ---: | ---: | --- |
| Entraînement | 1 402 023 | 266 507 | FTP-Patator, SSH-Patator, quatre DoS, Heartbleed |
| Validation | 456 752 | 2 216 | trois attaques web, Infiltration |
| Test | 414 322 | 288 923 | Bot, PortScan, DDoS |

Les familles du test sont absentes de l'entraînement. Celles de la validation aussi. La part d'attaques passe de 15,97 % dans l'entraînement à 0,48 % dans la validation et 41,08 % dans le test.

Si un fichier entier sert de test, ses familles d'attaque sont absentes de l'entraînement, puisqu'aucune famille n'occupe deux fichiers. Le lundi comme test ne contient aucune attaque : le rappel d'attaque n'y est pas définissable. Le lundi comme seul entraînement ne contient aucune attaque : les trois modèles ne peuvent pas apprendre la classe Attack.

Avantage : respecte la répartition par jour et évite qu'une campagne du vendredi serve à prédire le vendredi. Limite : ce n'est pas une évaluation des attaques apprises. Les trois modèles peuvent être ajustés dès que l'entraînement contient les deux classes, mais le test ne mesure pas les familles vues pendant l'apprentissage. Fuite restante : 33 563 profils bénins exacts existent dans plusieurs jours. Un découpage par jour peut donc placer la même ligne bénigne des deux côtés.

### B. Aléatoire stratifié, par contenus identiques

Règle simulée, indices non enregistrés : pour chaque étiquette, validation et test reçoivent chacun `n * 15 // 100` contenus exacts distincts, et l'entraînement reçoit le reste.

| Étiquette | Entraînement | Validation | Test |
| --- | ---: | ---: | ---: |
| BENIGN | 1 467 540 | 314 472 | 314 472 |
| DoS Hulk | 120 995 | 25 927 | 25 927 |
| PortScan | 63 575 | 13 622 | 13 622 |
| DDoS | 89 612 | 19 202 | 19 202 |
| DoS GoldenEye | 7 202 | 1 542 | 1 542 |
| FTP-Patator | 4 155 | 889 | 889 |
| SSH-Patator | 2 255 | 482 | 482 |
| DoS slowloris | 3 771 | 807 | 807 |
| DoS Slowhttptest | 3 660 | 784 | 784 |
| Bot | 1 369 | 292 | 292 |
| Web Attack, U+FFFD, Brute Force | 1 030 | 220 | 220 |
| Web Attack, U+FFFD, XSS | 458 | 97 | 97 |
| Infiltration | 26 | 5 | 5 |
| Web Attack, U+FFFD, Sql Injection | 15 | 3 | 3 |
| Heartbleed | 9 | 1 | 1 |
| **Total** | **1 765 672** | **378 345** | **378 345** |

Toutes les familles sont présentes dans les trois ensembles. Heartbleed n'y contribue qu'une ligne en validation et une en test. Les trois modèles peuvent être entraînés et évalués sur chaque famille.

Avantage : un contenu exact ne fuit pas d'un ensemble à l'autre, et la comparaison des modèles porte sur les mêmes types d'attaque. Limite : des flux seulement semblables, issus de la même campagne, peuvent encore se trouver des deux côtés. Ce découpage ne teste pas un jour futur. Il ne sépare pas non plus, à lui seul, les 698 profils contradictoires.

### C. Stratégie alternative

La stratégie C reprend B, avec trois compléments.

- Le groupe de découpage est le contenu des variables, pas seulement la ligne entière. Les 698 groupes contradictoires restent hors de l'entraînement et du test officiel.
- Le contrôle par jour de la stratégie A est calculé ensuite, sans servir à choisir le modèle.
- Aucun ordre horaire n'est inventé à l'intérieur d'un fichier.

Avantage : elle évite la fuite des copies exactes et des profils à deux étiquettes, tout en gardant une évaluation où les trois modèles voient chaque famille. Le contrôle par jour répond au cahier des charges sans devenir la métrique principale. Limite : écarter 7 020 lignes est un choix de qualité, pas une preuve que l'une des étiquettes est la bonne. Le contrôle par jour reste sévère et ne doit pas être mélangé au score principal.

## 4. Colonnes et anomalies

Traitements proposés pour un futur jeu dérivé. Rien n'est appliqué aux CSV bruts.

### Fwd Header Length

Les deux colonnes nommées ` Fwd Header Length`, positions 35 et 56, sont égales sur les 2 830 743 lignes. Aucune des deux n'a de valeur manquante. Proposition : conserver la première et ne pas recopier la seconde. La justification est l'identité constatée, pas une performance de test.

### Espaces dans les noms

Les noms ne sont pas normalisés dans les fichiers. Proposition : retirer les espaces de début et de fin dans le jeu dérivé, et écrire la correspondance dans le journal. `Flow Bytes/s` n'a pas d'espace initial. ` Flow Packets/s` en a un. Le nom du fichier jeudi après-midi, `Infilteration`, et l'étiquette `Infiltration` ne seraient pas unifiés.

### Valeurs manquantes et infinies

Toutes sont associées à ` Flow Duration` égale à 0.

| Situation | Lignes | Durée nulle |
| --- | ---: | ---: |
| `Flow Bytes/s` et ` Flow Packets/s` infinis positifs | 1 509 | 1 509 |
| `Flow Bytes/s` manquant et ` Flow Packets/s` infini positif | 1 358 | 1 358 |

Aucun infini négatif. Proposition : ne pas supprimer ces lignes. Dans le jeu dérivé, remplacer les infinis positifs par des valeurs manquantes. L'imputation médiane serait ajustée sur l'entraînement seulement, après le découpage. La durée nulle resterait visible dans ` Flow Duration`.

### U+FFFD

Les trois étiquettes web contiennent réellement les octets UTF-8 `EF BF BD`, soit U+FFFD, entre `Web Attack` et le type. Proposition : les garder comme trois familles distinctes, sous leur forme lue. Ne pas deviner un tiret et ne pas les fusionner. Pour la classe binaire, elles restent des attaques. Le fichier brut n'est pas réécrit.
