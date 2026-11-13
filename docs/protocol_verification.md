# Vérification avant nettoyage

Contrôle du 13 novembre 2026, en lecture seule. Aucune ligne n'a été supprimée. Les CSV bruts n'ont pas été modifiés. Aucun découpage n'a été enregistré. Aucun modèle n'a été ajusté. Aucun calcul SHAP n'a été lancé.

Les trois algorithmes restent la régression logistique, la forêt aléatoire et XGBoost. Le protocole visé, encore non appliqué, est l'apprentissage lundi–mercredi, la validation jeudi et le test vendredi. Les chiffres viennent de `scripts/verify_protocol_rules.py`, de `outputs/metrics/protocol_verification.json`, et d'un contrôle complémentaire sur les jetons `NaN`.

## 1. Choix du modèle sur la validation du jeudi

Les trois modèles n'ont pas de PR-AUC dans ce rapport. En produire un demanderait de les entraîner. Cette phase l'interdit, et un score simulé ne serait pas une mesure.

Le futur classement utilisera les mêmes lignes pour les trois modèles. L'ordre déjà fixé est celui-ci.

1. Les hyperparamètres se choisissent par validation croisée interne sur un sous-échantillon de l'apprentissage, avec l'average precision des scores continus. La validation officielle n'entre pas dans cette recherche. Le test n'y entre pas.
2. Le modèle principal se choisit ensuite, parmi les trois modèles réajustés sur tout l'apprentissage, par le PR-AUC de la validation. Le PR-AUC est `average_precision` sur `predict_proba` de la classe Attack. Les prédictions binaires ne servent pas à ce calcul.
3. Le seuil se choisit après, sur la validation du modèle déjà retenu, en minimisant `1 × FPR + 2 × FNR`. Le seuil 0,5 est aussi rapporté. Ce choix ne rouvre ni le modèle ni les hyperparamètres.
4. Le test est lu une fois, pour l'évaluation. Il ne choisit ni le modèle, ni les hyperparamètres, ni le seuil.

Le choix du modèle et le choix du seuil sont deux décisions successives, sur deux critères différents. Le PR-AUC classe les modèles. Le critère FPR + 2 FNR place le seuil du modèle classé premier.

### Ce que la validation permet de stabiliser

Avant toute règle, le jeudi contient 458 968 lignes, dont 2 216 attaques (0,48 %). Après conservation de la première occurrence exacte, il en reste 398 717, dont 2 179 attaques (0,55 %). L'exclusion des profils contradictoires retire ensuite 607 lignes normales et aucune attaque : 398 110 lignes, toujours 2 179 attaques (0,55 %).

| Famille du jeudi | Avant | Après les deux règles | Part des attaques conservées |
| --- | ---: | ---: | ---: |
| Web Attack U+FFFD Brute Force | 1 507 | 1 470 | 67,46 % |
| Web Attack U+FFFD XSS | 652 | 652 | 29,92 % |
| Web Attack U+FFFD Sql Injection | 21 | 21 | 0,96 % |
| Infiltration | 36 | 36 | 1,65 % |

Ces quatre familles sont absentes de l'apprentissage lundi–mercredi. Un tirage avec remise de la validation conservée contient en moyenne 2 179 attaques, avec un écart-type de 46,6 si les lignes étaient indépendantes. L'effectif positif varie donc d'environ 2 %. Cette valeur est un plancher : les attaques du jeudi viennent d'une même campagne, et les lignes ne sont pas indépendantes. L'écart réel d'un PR-AUC sera plus grand. Il ne peut être mesuré qu'après l'entraînement, par un bootstrap des scores de validation, sans regarder le test.

Avec 395 931 lignes normales et 2 179 attaques, un taux de faux positifs de 0,550 % produit déjà autant de fausses alertes que d'attaques. La précision, et donc le PR-AUC, bouge dès qu'un modèle se trompe sur une fraction de pourcent du trafic normal.

### Limites du choix du modèle sur cette validation

Le PR-AUC binaire du jeudi est d'abord un score de Web Attack Brute Force et de XSS. Les 21 injections SQL et les 36 infiltrations ne départagent pas les trois modèles.

Ce score mesure le transfert vers des familles absentes de l'apprentissage. Il ne mesure pas la reconnaissance des quatre DoS, de FTP-Patator, de SSH-Patator ni de Heartbleed, puisque ces familles ne reviennent pas le jeudi.

Bot, PortScan et DDoS sont absents de l'apprentissage et absents de la validation. Le modèle retenu le jeudi n'est pas, pour cette raison, le modèle qui aurait été retenu le vendredi. Vérifier cette coïncidence lirait le test pendant la sélection. Cette lecture n'est pas faite.

Les trois modèles restent tous évalués. Le choix d'un modèle principal, lorsqu'il sera autorisé, restera un choix de validation. Ses chiffres de sélection ne seront pas présentés comme une estimation impartiale. L'estimation impartiale sera le test, lu ensuite, une seule fois.

## 2. Identification des profils contradictoires

La règle proposée utilise le MD5 des octets situés avant la dernière virgule. Elle a été rejouée avec le parseur `csv` de Python, qui lit les 78 colonnes de caractéristiques, puis avec les valeurs numériques que pandas obtient de ces colonnes.

Les huit fichiers ont le même en-tête : 79 colonnes, dernière colonne ` Label`. Sur les 2 830 743 lignes :

| Contrôle | Résultat |
| --- | ---: |
| Lignes contenant un guillemet CSV | 0 |
| Lignes dont le nombre de virgules n'est pas 78 | 0 |
| Lignes dont le parseur ne rend pas 79 champs | 0 |
| Lignes dont les 78 champs relus diffèrent des octets avant la dernière virgule | 0 |
| Lignes dont l'étiquette relue diffère du dernier champ brut | 0 |
| Champs qui ne sont pas un nombre, vide ou `NaN` | 0 |

Sans guillemet et avec exactement 78 virgules, la dernière virgule est bien le séparateur de l'étiquette. Le MD5 des octets et le MD5 des 78 colonnes lues par `csv` décrivent la même partition. Les deux méthodes retrouvent 698 profils et 7 020 lignes.

| Étiquette | Lignes |
| --- | ---: |
| DoS Hulk | 5 224 |
| PortScan | 1 053 |
| BENIGN | 736 |
| DDoS | 6 |
| DoS slowloris | 1 |
| Total | 7 020 |

| Jour | Étiquette | Lignes |
| --- | --- | ---: |
| Lundi | BENIGN | 3 |
| Mardi | BENIGN | 33 |
| Mercredi | DoS Hulk | 5 224 |
| Mercredi | BENIGN | 49 |
| Mercredi | DoS slowloris | 1 |
| Jeudi matin | BENIGN | 9 |
| Jeudi après-midi | BENIGN | 620 |
| Vendredi matin | BENIGN | 10 |
| Vendredi PortScan | PortScan | 1 053 |
| Vendredi PortScan | BENIGN | 7 |
| Vendredi DDoS | DDoS | 6 |
| Vendredi DDoS | BENIGN | 5 |

Les 7 020 lignes ont été relues. Dans chacun des 698 groupes, les octets des 78 caractéristiques sont identiques d'une ligne à l'autre, et le groupe porte deux étiquettes. Aucun groupe ne fait exception.

L'échantillon pandas compte 800 lignes, 100 par fichier, tirées par réservoir avec la graine 42. `read_csv(..., dtype=str)` rend les mêmes 79 champs que le parseur `csv`, et la lecture numérique rend les mêmes valeurs `float` que `float()` sur ces champs. Écarts : 0.

Un second échantillon prend, parmi les 7 020 lignes, jusqu'à deux lignes par jour et par étiquette : 23 lignes. La lecture en chaînes coïncide aussi. La comparaison numérique automatique signalait 3 cellules. Ce sont trois fois le texte `NaN` dans `Flow Bytes/s`, que pandas lit comme valeur manquante. `float("NaN")` est cette même valeur manquante. Le compteur traitait seulement le champ vide comme manquant, donc il les avait marquées à tort.

Sur tout le corpus, le seul jeton manquant des caractéristiques est le texte `NaN`, 1 358 fois. Il n'y a pas de champ vide. En remplaçant ce jeton par une valeur manquante puis en comparant les vecteurs `float64`, on obtient encore 2 521 664 profils distincts, 698 profils contradictoires et 7 020 lignes. La conversion numérique ne fusionne aucun profil supplémentaire et n'en écarte aucun.

La règle peut donc s'énoncer sur les colonnes lues : deux lignes ont le même profil si leurs 78 caractéristiques, parsées comme CSV, sont identiques. Si ce profil reçoit plusieurs étiquettes, tout le groupe sort de l'apprentissage, de la validation et du test officiels. Le MD5 avant la dernière virgule est une mise en œuvre exacte de cette comparaison sur ces huit fichiers. Aucune étiquette n'est choisie. Aucun CSV n'est réécrit.

## 3. Première occurrence d'un doublon exact

Un doublon exact est la ligne entière, étiquette comprise. La règle conserve la première occurrence dans l'ordre des fichiers, puis dans l'ordre des lignes du fichier. Les copies ultérieures ne sont pas reprises.

Avant la règle, 33 563 contenus exacts apparaissent dans plusieurs fichiers. Tous sont `BENIGN` : aucun contenu d'attaque exact n'occupe deux fichiers. Parmi eux, 28 024 contenus occupent au moins deux des trois ensembles lundi–mercredi, jeudi et vendredi. Les 5 539 autres restent à l'intérieur d'un seul ensemble, parce que leurs fichiers appartiennent au même bloc.

Le contrôle a parcouru les 2 522 362 contenus distincts. Pour chacun, le fichier retenu est le plus ancien fichier où il apparaît. Après la règle, le nombre de contenus présents dans plusieurs ensembles est 0. Chaque contenu conservé a exactement une ligne, dans un seul ensemble.

| Ensemble | Classe | Avant | Après la première occurrence | Après la première occurrence et l'exclusion des 698 profils |
| --- | --- | ---: | ---: | ---: |
| Apprentissage | Lignes | 1 668 530 | 1 527 339 | 1 527 133 |
| Apprentissage | Normal | 1 402 023 | 1 324 428 | 1 324 353 |
| Apprentissage | Attack | 266 507 | 202 911 | 202 780 |
| Validation | Lignes | 458 968 | 398 717 | 398 110 |
| Validation | Normal | 456 752 | 396 538 | 395 931 |
| Validation | Attack | 2 216 | 2 179 | 2 179 |
| Test | Lignes | 703 245 | 596 306 | 595 723 |
| Test | Normal | 414 322 | 375 518 | 375 502 |
| Test | Attack | 288 923 | 220 788 | 220 221 |

La première occurrence ramène le corpus à 2 522 362 lignes, soit le nombre de contenus exacts distincts. Les 308 381 copies supplémentaires sont retirées, et rien d'autre. L'exclusion des profils contradictoires retire ensuite 1 396 contenus, deux par groupe, et laisse 2 520 966 lignes. Ces deux totaux étaient déjà connus ; ils se retrouvent ici comme somme des trois ensembles.

Familles d'attaque après la première occurrence :

| Famille | Apprentissage | Validation | Test |
| --- | ---: | ---: | ---: |
| FTP-Patator | 5 933 | 0 | 0 |
| SSH-Patator | 3 219 | 0 | 0 |
| DoS slowloris | 5 385 | 0 | 0 |
| DoS Slowhttptest | 5 228 | 0 | 0 |
| DoS Hulk | 172 849 | 0 | 0 |
| DoS GoldenEye | 10 286 | 0 | 0 |
| Heartbleed | 11 | 0 | 0 |
| Web Attack U+FFFD Brute Force | 0 | 1 470 | 0 |
| Web Attack U+FFFD XSS | 0 | 652 | 0 |
| Web Attack U+FFFD Sql Injection | 0 | 21 | 0 |
| Infiltration | 0 | 36 | 0 |
| Bot | 0 | 0 | 1 953 |
| PortScan | 0 | 0 | 90 819 |
| DDoS | 0 | 0 | 128 016 |

Aucune des quatorze familles ne disparaît, ni du corpus ni de son ensemble. Heartbleed reste à 11, l'injection SQL à 21, l'infiltration à 36. L'exclusion des 698 profils retire des exemplaires de DoS Hulk, DoS slowloris, PortScan et DDoS, sans vider ces familles : il en reste respectivement 172 719, 5 384, 90 255 et 128 013.

## État

Les trois vérifications soutiennent les règles proposées. Le MD5 avant la dernière virgule identifie les mêmes 698 profils que les colonnes CSV et que leurs valeurs numériques. La première occurrence empêche un contenu exact de figurer dans deux ensembles et conserve toutes les familles. La validation du jeudi peut porter un PR-AUC, avec les limites de la section 1 : peu d'attaques, familles absentes de l'apprentissage, et un classement qui ne se transporte pas au vendredi.

Le nettoyage n'est pas commencé. `config/config.yaml` n'a pas été modifié. J'attends une validation explicite avant de supprimer une ligne, d'écrire un découpage ou d'entraîner les trois modèles.
