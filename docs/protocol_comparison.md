# Comparaison des protocoles

Analyse du 13 novembre 2026, en lecture seule. Aucune ligne n'a été supprimée. Les CSV bruts n'ont pas été modifiés. Aucun modèle n'a été ajusté. Aucun découpage n'a été enregistré. Aucun calcul SHAP n'a été lancé.

Les trois algorithmes restent la régression logistique, la forêt aléatoire et XGBoost.

Le protocole principal recommandé, en attente de validation, est le découpage chronologique contigu : apprentissage lundi–mercredi, validation jeudi (matin et après-midi), test vendredi (matin, PortScan, DDoS). Le protocole complémentaire recommandé est le découpage groupé stratifié 70 % / 15 % / 15 % sur les profils distincts, après mise à l'écart des profils contradictoires.

Les effectifs viennent de `scripts/compare_protocols.py` et de `outputs/metrics/protocol_comparison.json`. Chaque total d'un découpage vaut 2 830 743 lignes.

## Ordre chronologique retenu

Les fichiers n'ont pas de colonne de date ni d'heure. L'ordre utilisé est celui des noms de fichiers et du calendrier publié de CIC-IDS2017. Jeudi et vendredi sont déjà coupés en plusieurs fichiers. Les cinq jours donnent donc huit blocs, dans cet ordre :

1. Lundi
2. Mardi
3. Mercredi
4. Jeudi matin (attaques web)
5. Jeudi après-midi (Infiltration)
6. Vendredi matin (Bot)
7. Vendredi après-midi PortScan
8. Vendredi après-midi DDoS

Les 21 répartitions étudiées sont toutes les coupes contigues de cette suite : l'apprentissage est un préfixe, la validation un bloc du milieu, le test un suffixe. Aucune répartition ne place un jour tardif dans l'apprentissage et un jour antérieur dans le test.

Chaque famille d'attaque n'occupe qu'un seul fichier. Dans chacune des 21 répartitions, toute famille présente dans l'apprentissage est absente de la validation et absente du test. Toute famille du test est absente de l'apprentissage. C'est une propriété des fichiers, pas un accident de coupe.

## 1. Découpages chronologiques

Lecture commune des fuites. Une copie exacte est la ligne entière, étiquette comprise. Le contrôle déjà publié a montré que les copies exactes qui traversent deux fichiers sont toutes `BENIGN`. Les intersections ci-dessous comptent ces lignes. La colonne « lignes du test » est le nombre de lignes du test dont le contenu exact existe déjà dans l'apprentissage.

Un profil identique est le contenu des variables, sans l'étiquette. Les 698 profils qui portent deux étiquettes sont détaillés à la section 3. Leur traversée d'une coupe est indiquée pour les répartitions discutées ensuite.

### Apprentissage limité au lundi

L'apprentissage contient 529 918 lignes, toutes `BENIGN`. La classe Attack est absente. La régression logistique, la forêt aléatoire et XGBoost ne peuvent pas apprendre à séparer les deux classes. `scale_pos_weight` ne peut pas être calculé. Ces six répartitions sont écartées, quel que soit le test.

| # | Validation | Test | Validation : lignes, attaques | Test : lignes, attaques | Familles du test | Copies exactes apprentissage → test |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Mardi | Mercredi à vendredi DDoS | 445 909, 3,10 % | 1 854 916, 29,32 % | 12 | 329 contenus, 5 802 lignes |
| 2 | Mardi–mercredi | Jeudi à vendredi DDoS | 1 138 612, 23,41 % | 1 162 213, 25,05 % | 7 | 275 contenus, 3 799 lignes |
| 3 | Mardi à jeudi matin | Jeudi après-midi à vendredi DDoS | 1 308 978, 20,53 % | 991 847, 29,13 % | 4 | 238 contenus, 3 080 lignes |
| 4 | Mardi à jeudi | Vendredi entier | 1 597 580, 16,82 % | 703 245, 41,08 % | 3 | 196 contenus, 1 554 lignes |
| 5 | Mardi à vendredi matin | Vendredi PortScan et DDoS | 1 788 613, 15,13 % | 512 212, 56,02 % | 2 | 128 contenus, 780 lignes |
| 6 | Mardi à vendredi PortScan | Vendredi DDoS | 2 075 080, 20,70 % | 225 745, 56,71 % | 1 | 61 contenus, 182 lignes |

Le test de la répartition 1 contient les douze familles à partir du mercredi. Celui des répartitions 2 à 6 commence plus tard : les familles des fichiers déjà placés en validation n'y sont plus. Dans les six cas, l'apprentissage ne contient aucune famille.

### Apprentissage lundi–mardi

L'apprentissage contient 975 827 lignes : 961 992 Normal (98,58 %) et 13 835 Attack (1,42 %). Les seules familles sont FTP-Patator et SSH-Patator. Les trois modèles peuvent être ajustés. Le concept d'attaque appris est limité au brute force. FTP-Patator et SSH-Patator sont absents de la validation et du test dans les cinq répartitions.

| # | Validation | Attaques en validation | Test | Attaques au test | Familles du test | Copies exactes apprentissage → test |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 7 | Mercredi | 252 672 (36,48 %) | Jeudi à vendredi DDoS | 291 139 (25,05 %) | Web, Infiltration, Bot, PortScan, DDoS | 11 869 contenus, 30 416 lignes |
| 8 | Mercredi et jeudi matin | 254 852 (29,53 %) | Jeudi après-midi à vendredi DDoS | 288 959 (29,13 %) | Infiltration, Bot, PortScan, DDoS | 10 167 contenus, 23 919 lignes |
| 9 | Mercredi et jeudi | 254 888 (22,13 %) | Vendredi entier | 288 923 (41,08 %) | Bot, PortScan, DDoS | 7 813 contenus, 14 978 lignes |
| 10 | Mercredi à vendredi matin | 256 854 (19,13 %) | Vendredi PortScan et DDoS | 286 957 (56,02 %) | PortScan, DDoS | 4 866 contenus, 7 788 lignes |
| 11 | Mercredi à vendredi PortScan | 415 784 (25,52 %) | Vendredi DDoS | 128 027 (56,71 %) | DDoS | 1 990 contenus, 2 585 lignes |

La validation est large. Le test aussi, sauf la répartition 11, réduite au DDoS. La limite commune est l'apprentissage : 13 835 attaques de brute force pour près d'un million de flux normaux. Une forêt ou XGBoost peut séparer ces patrons ; l'évaluation des jours suivants mesure alors un transfert depuis FTP-Patator et SSH-Patator seulement.

### Apprentissage lundi–mercredi

L'apprentissage est le même pour les répartitions 12 à 15 : 1 668 530 lignes, 1 402 023 Normal (84,03 %) et 266 507 Attack (15,97 %). Familles : FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed. Ces sept familles sont absentes de la validation et du test. Les trois modèles peuvent être ajustés avec une classe Attack réelle. `scale_pos_weight` vaut le rapport Normal / Attack de cet apprentissage.

| # | Validation | Lignes | Normal | Attack | Familles de validation | Test | Familles du test | Copies exactes apprentissage → test |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| 12 | Jeudi matin | 170 366 | 168 186 | 2 180 (1,28 %) | trois attaques web | Jeudi après-midi et vendredi | Infiltration, Bot, PortScan, DDoS | 20 202 contenus, 46 262 lignes |
| 13 | Jeudi entier | 458 968 | 456 752 | 2 216 (0,48 %) | trois attaques web, Infiltration | Vendredi entier, 703 245 lignes, 288 923 attaques (41,08 %) | Bot, PortScan, DDoS | 15 233 contenus, 29 084 lignes |
| 14 | Jeudi et vendredi matin | 650 001 | 645 819 | 4 182 (0,64 %) | trois attaques web, Infiltration, Bot | PortScan et DDoS, 512 212 lignes, 286 957 attaques (56,02 %) | PortScan, DDoS | 9 126 contenus, 14 888 lignes |
| 15 | Jeudi à vendredi PortScan | 936 468 | 773 356 | 163 112 (17,42 %) | trois attaques web, Infiltration, Bot, PortScan | DDoS, 225 745 lignes, 128 027 attaques (56,71 %) | DDoS | 3 720 contenus, 4 930 lignes |

La répartition 13 est la seule de ce bloc qui réserve le vendredi entier au test. Son test contient trois familles jamais vues, et 288 923 attaques : le rappel d'attaque et le PR-AUC y sont définis pour les trois modèles. Sa validation ne contient que 2 216 attaques (0,48 %), de familles qui n'apparaissent pas le vendredi. Le seuil fixé sur cette validation est un point de fonctionnement fragile. Le classement des trois modèles sur ce protocole s'appuie sur le PR-AUC des scores continus du vendredi. Le seuil de validation et le seuil 0,5 sont tous les deux à rapporter. Aucun des deux n'est choisi sur le test.

La répartition 12 isole le jeudi matin : 2 180 attaques web, et le test mélange l'infiltration (36 lignes) avec le vendredi. La répartition 14 ajoute le vendredi matin à la validation et retire Bot du test. La répartition 15 donne une validation large, au prix d'un test réduit au seul DDoS. Le seuil y serait estimable. L'évaluation officielle ne porterait plus que sur une famille.

Copies exactes de la répartition 13, en plus de l'intersection apprentissage → test : 14 490 contenus de l'apprentissage réapparaissent dans 30 141 lignes du jeudi ; 10 409 contenus du jeudi réapparaissent dans 23 288 lignes du vendredi. Ces deux derniers ensembles se recouvrent avec l'intersection apprentissage → test. Ils ne s'additionnent pas.

Profils contradictoires qui traversent la répartition 13 : 633 groupes sur 698 ont leurs deux étiquettes dans des ensembles différents. 564 de ces groupes sont `BENIGN` le jeudi après-midi et PortScan le vendredi. Ils traversent donc la validation et le test. Vingt groupes portent DoS Hulk le mercredi et `BENIGN` le vendredi : ils traversent l'apprentissage et le test. Sans la règle de la section 3, le jeudi enseignerait comme normaux des profils que le vendredi étiquette PortScan.

### Apprentissage jusqu'au jeudi matin

L'apprentissage contient 1 838 896 lignes, 1 570 209 Normal et 268 687 Attack (14,61 %). Les familles ajoutées sont les trois attaques web. Heartbleed, les quatre DoS et les deux Patator restent dans l'apprentissage, donc absents de la validation et du test.

| # | Validation | Attack en validation | Test | Attack au test | Familles du test | Copies exactes apprentissage → test |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| 16 | Jeudi après-midi | 36 (0,01 %) | Vendredi entier | 288 923 (41,08 %) | Bot, PortScan, DDoS | 17 324 contenus, 31 863 lignes |
| 17 | Jeudi après-midi et vendredi matin | 2 002 (0,42 %) | PortScan et DDoS | 286 957 (56,02 %) | PortScan, DDoS | 10 253 contenus, 16 232 lignes |
| 18 | Jeudi après-midi à vendredi PortScan | 160 932 (21,01 %) | DDoS | 128 027 (56,71 %) | DDoS | 4 149 contenus, 5 398 lignes |

La répartition 16 place 36 lignes Infiltration dans la validation. Un seuil sur 36 positifs n'est pas un réglage. La répartition 18 rend le seuil estimable et réduit le test au DDoS.

### Apprentissage jusqu'au jeudi, ou jusqu'au vendredi matin

| # | Apprentissage | Validation | Test | Copies exactes apprentissage → test | Groupes contradictoires à cheval |
| --- | --- | --- | --- | ---: | ---: |
| 19 | Lundi à jeudi : 2 127 498 lignes, 268 723 attaques (12,63 %). Familles : deux Patator, quatre DoS, Heartbleed, trois web, Infiltration | Vendredi matin : 191 033 lignes, 1 966 Bot (1,03 %) | PortScan et DDoS : 512 212 lignes, 286 957 attaques (56,02 %) | 11 641 contenus, 18 298 lignes | 587, dont 577 entre apprentissage et test |
| 20 | Même apprentissage | Vendredi matin et PortScan : 477 500 lignes, 160 896 attaques (33,70 %), Bot et PortScan | DDoS : 225 745 lignes, 128 027 attaques (56,71 %) | 4 841 contenus, 6 649 lignes | 587, surtout entre apprentissage et validation |
| 21 | Lundi à vendredi matin : 2 318 531 lignes, 270 689 attaques (11,68 %). Bot s'ajoute à l'apprentissage | PortScan : 286 467 lignes, 158 930 attaques (55,48 %) | DDoS : 225 745 lignes, 128 027 attaques (56,71 %) | 5 112 contenus, 6 936 lignes | 578, dont 570 entre apprentissage et validation ; aucun entre validation et test |

Les trois modèles peuvent être ajustés sur ces trois répartitions. La 19 évalue deux grandes familles jamais vues, avec une validation de 1 966 Bot seulement. La 20 et la 21 donnent une validation assez grande pour le seuil FPR + 2 FNR. Leur test officiel est le seul DDoS. Dans la 21, les 158 930 attaques PortScan servent à choisir le seuil, pas à l'évaluation finale. Les familles de l'apprentissage, y compris Bot, sont absentes du test.

### Pertinence pour les trois modèles

Les trois modèles produisent des scores continus dès que l'apprentissage contient les deux classes. Le PR-AUC se calcule sur ces scores. Cela couvre les répartitions 7 à 21.

Le seuil demandé, minimisant FPR + 2 FNR sur la validation seule, demande des attaques en validation. En dessous de quelques milliers de positifs, ce seuil bouge fortement d'un échantillon à l'autre. C'est le cas des répartitions 12, 13, 14, 16, 17 et 19. Les répartitions 15, 18, 20 et 21 ont une validation large, et un test réduit au DDoS.

Le rappel d'attaque n'est définissable que si le test contient des attaques. C'est vrai des 21 répartitions. Il n'est pas définissable famille par famille pour les familles restées dans l'apprentissage.

## 2. Découpage groupé stratifié

Ce protocole reste un complément possible. Il n'est pas appliqué. La simulation porte sur les contenus distincts de variables et d'étiquette, après retrait des 698 profils contradictoires. Il resterait 2 520 966 contenus. Pour chaque étiquette, la validation et le test recevraient chacun `n * 15 // 100` contenus, et l'apprentissage le reste. Les indices ne sont pas enregistrés.

| Étiquette | Contenus | Apprentissage | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| BENIGN | 2 095 786 | 1 467 052 | 314 367 | 314 367 |
| DoS Hulk | 172 719 | 120 905 | 25 907 | 25 907 |
| DDoS | 128 013 | 89 611 | 19 201 | 19 201 |
| PortScan | 90 255 | 63 179 | 13 538 | 13 538 |
| DoS GoldenEye | 10 286 | 7 202 | 1 542 | 1 542 |
| FTP-Patator | 5 933 | 4 155 | 889 | 889 |
| DoS slowloris | 5 384 | 3 770 | 807 | 807 |
| DoS Slowhttptest | 5 228 | 3 660 | 784 | 784 |
| SSH-Patator | 3 219 | 2 255 | 482 | 482 |
| Bot | 1 953 | 1 369 | 292 | 292 |
| Web Attack U+FFFD Brute Force | 1 470 | 1 030 | 220 | 220 |
| Web Attack U+FFFD XSS | 652 | 458 | 97 | 97 |
| Infiltration | 36 | 26 | 5 | 5 |
| Web Attack U+FFFD Sql Injection | 21 | 15 | 3 | 3 |
| Heartbleed | 11 | 9 | 1 | 1 |
| Total | 2 520 966 | 1 764 696 | 378 135 | 378 135 |

L'apprentissage contient 1 467 052 Normal et 297 644 Attack (16,87 %). La validation et le test contiennent chacun 314 367 Normal et 63 768 Attack (16,86 %). Les quinze familles sont présentes dans les trois ensembles. Un même profil de variables ne traverse pas deux ensembles, parce que le groupe de découpage est ce profil et que les profils à deux étiquettes sont sortis avant le tirage.

Les trois modèles peuvent être entraînés et évalués. Le seuil de validation et le test ont la même composition de familles et la même proportion d'attaques. Heartbleed n'a qu'une ligne en test, l'injection SQL trois, l'infiltration cinq : le rappel par famille de ces trois étiquettes reste instable. Le score binaire, lui, porte sur 63 768 attaques.

### Pourquoi ce découpage ne tient pas lieu d'évaluation chronologique

Le cahier des charges demande une évaluation chronologique lorsque les données sont organisées par jour. CIC-IDS2017 place chaque famille sur un seul jour. Un détecteur appris du lundi au jeudi n'a pas vu Bot, PortScan ni DDoS.

Le découpage groupé répond à une autre question : le modèle reconnaît-il des familles qu'on lui a déjà montrées, sans qu'une ligne identique passe de l'apprentissage au test. Chaque famille est des deux côtés. Un PR-AUC élevé dans ce protocole peut venir de signatures vues pendant l'apprentissage. Le même modèle peut échouer le vendredi, sur des familles absentes de l'apprentissage.

Les deux résultats peuvent donc diverger. Publier seulement le score groupé présenterait comme évaluation temporelle une évaluation où l'avenir a déjà été montré au modèle. Le groupé reste utile à côté, parce que le découpage par jour ne permet pas de mesurer la détection des familles apprises : elles ne reviennent dans aucun jour ultérieur.

Les choix faits dans un protocole restent dans ce protocole. Le seuil, le modèle et les variables du rapport chronologique se décident sur la validation chronologique. Ceux du rapport groupé se décident sur la validation groupée.

## 3. Les 7 020 lignes contradictoires

Un profil est l'empreinte MD5 des octets de la ligne avant la dernière virgule. L'étiquette est le champ UTF-8 après cette virgule. Le contrôle retrouve 698 profils associés à deux étiquettes, et 7 020 lignes. Aucun profil n'a plus de deux étiquettes. Les conserver comme doublons exacts ne les retire pas : les deux lignes diffèrent par l'étiquette.

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

Les 698 groupes se répartissent en quatre paires, déjà mesurées, et en 20 situations de jours.

| Paire | Groupes | Situation dominante |
| --- | ---: | --- |
| BENIGN et PortScan | 564 | BENIGN le jeudi après-midi, PortScan le vendredi. Fichiers disjoints. |
| BENIGN et DoS Hulk | 130 | DoS Hulk toujours le mercredi. Le côté BENIGN est le mercredi seul pour 34 groupes, et un ou plusieurs autres jours pour les 96 autres. |
| BENIGN et DDoS | 3 | BENIGN un jour antérieur, DDoS le vendredi après-midi. Fichiers disjoints. |
| BENIGN et DoS slowloris | 1 | DoS slowloris le mercredi, BENIGN le mercredi et le jeudi après-midi. |

| Groupes | Étiquette A | Jours A | Étiquette B | Jours B |
| ---: | --- | --- | --- | --- |
| 564 | BENIGN | Jeudi après-midi | PortScan | Vendredi PortScan |
| 34 | BENIGN | Mercredi | DoS Hulk | Mercredi |
| 33 | BENIGN | Jeudi après-midi | DoS Hulk | Mercredi |
| 22 | BENIGN | Mardi | DoS Hulk | Mercredi |
| 9 | BENIGN | Vendredi matin | DoS Hulk | Mercredi |
| 6 | BENIGN | Jeudi matin | DoS Hulk | Mercredi |
| 6 | BENIGN | Mardi et mercredi | DoS Hulk | Mercredi |
| 5 | BENIGN | Mercredi et jeudi après-midi | DoS Hulk | Mercredi |
| 3 | BENIGN | Vendredi PortScan | DoS Hulk | Mercredi |
| 3 | BENIGN | Vendredi DDoS | DoS Hulk | Mercredi |
| 2 | BENIGN | Lundi | DoS Hulk | Mercredi |
| 2 | BENIGN | Mardi et vendredi PortScan | DoS Hulk | Mercredi |
| 1 | BENIGN | Jeudi matin et vendredi PortScan | DoS Hulk | Mercredi |
| 1 | BENIGN | Mercredi et jeudi matin | DoS Hulk | Mercredi |
| 1 | BENIGN | Vendredi matin | DDoS | Vendredi DDoS |
| 1 | BENIGN | Mardi et jeudi après-midi | DoS Hulk | Mercredi |
| 1 | BENIGN | Lundi, mercredi et vendredi DDoS | DoS Hulk | Mercredi |
| 1 | BENIGN | Jeudi matin | DDoS | Vendredi DDoS |
| 1 | BENIGN | Jeudi après-midi et vendredi DDoS | DoS Hulk | Mercredi |
| 1 | BENIGN | Jeudi après-midi | DDoS | Vendredi DDoS |
| 1 | BENIGN | Mercredi et jeudi après-midi | DoS slowloris | Mercredi |

### Règle proposée, non appliquée

La règle tient en quatre points, reproductibles sans réécrire les CSV.

1. La clé de profil est le MD5 des octets de la ligne brute avant la dernière virgule. L'étiquette est le dernier champ, décodé en UTF-8.
2. Si une clé possède au moins deux étiquettes distinctes dans les huit fichiers, le groupe entier est contradictoire.
3. Les lignes de ce groupe sont exclues de l'apprentissage officiel, de la validation officielle et du test officiel. Aucune étiquette n'est élue. Aucune ligne n'est réétiquetée.
4. Le manifeste de comptage est celui de cette section : 698 groupes, 7 020 lignes, par étiquette, par jour et par situation. Il reste dans le rapport. Il ne sert pas à l'entraînement.

Sur la répartition 13, cette règle retirerait 5 310 lignes de l'apprentissage (5 224 DoS Hulk, 1 DoS slowloris, 85 BENIGN), 629 lignes BENIGN de la validation, et 1 081 lignes du test (1 053 PortScan, 6 DDoS, 22 BENIGN). Les effectifs qui resteraient seraient :

| Ensemble | Lignes | Normal | Attack |
| --- | ---: | ---: | ---: |
| Apprentissage lundi–mercredi | 1 663 220 | 1 401 938 | 261 282 |
| Validation jeudi | 458 339 | 456 123 | 2 216 |
| Test vendredi | 702 164 | 414 300 | 287 864 |

Ces trois totaux font 2 823 723, soit 2 830 743 moins 7 020. Ils ne sont pas enregistrés comme découpage.

Une seconde règle, distincte, concerne les copies exactes bénignes qui traversent les jours. Pour le protocole chronologique, une ligne exacte déjà vue dans un fichier antérieur ne serait pas reprise dans un ensemble plus tardif. L'exemplaire conservé est celui du premier fichier. Sur la répartition 13, 29 084 lignes du vendredi ont un contenu exact déjà présent du lundi au mercredi, et 23 288 lignes du vendredi ont un contenu exact déjà présent le jeudi. Le recouvrement entre ces deux ensembles n'a pas été recompté : les deux nombres restent séparés. Cette règle n'est pas appliquée non plus.

## 4. Recommandation

### Protocole principal

Découpage chronologique contigu, répartition 13.

- Apprentissage : lundi, mardi, mercredi.
- Validation : jeudi matin et jeudi après-midi, tenus ensemble.
- Test : vendredi matin, vendredi PortScan et vendredi DDoS, tenus ensemble.
- Avant ce découpage, lorsque la phase sera ouverte : appliquer la règle des profils contradictoires, puis ne garder une copie exacte que dans le premier fichier où elle apparaît.
- Comparaison des trois modèles : PR-AUC sur les scores continus du test. Seuil choisi seulement sur la validation par FPR + 2 FNR, et seuil 0,5 rapporté à côté. Le test ne choisit ni le seuil, ni le modèle, ni les variables.

Avantages. L'ordre des jours est respecté. L'apprentissage contient les deux classes et sept familles, donc les trois modèles peuvent être ajustés. Le test est le dernier jour entier, avec 288 923 attaques de trois familles absentes de l'apprentissage. Aucune attaque du vendredi ne sert à régler le modèle. Le PR-AUC du vendredi ne dépend pas du seuil fragile du jeudi.

Limites. Le test ne mesure pas les familles apprises : elles ne reviennent pas. La validation n'a que 2 216 attaques, d'autres familles, et une proportion de 0,48 %. Le seuil qui en sort est un point de fonctionnement transporté vers un autre mélange d'attaques. Il est publié comme tel. Les proportions d'attaque passent de 15,97 % à 0,48 % puis à 41,08 %. Des copies exactes bénignes traversent les jours tant que la seconde règle n'est pas appliquée. 564 profils sont normaux le jeudi et PortScan le vendredi tant que la première règle n'est pas appliquée.

### Protocole complémentaire

Découpage groupé stratifié décrit à la section 2, sur les contenus qui restent après la même exclusion des 698 profils. Graine et indices seulement après validation. Ce protocole a son propre seuil, choisi sur sa propre validation.

Avantages. Les quinze familles sont dans les trois ensembles. La proportion d'attaque est 16,87 % à l'apprentissage et 16,86 % en validation et en test. Un profil identique ne traverse pas deux ensembles. Les trois modèles y sont comparables sur les familles qu'ils ont vues. Le seuil et le test portent sur le même mélange.

Limites. Ce score n'est pas une évaluation du lendemain. Les familles rares ont une à cinq lignes de test. Des flux du même jour, proches sans être identiques, peuvent encore se trouver des deux côtés : le blocage porte sur le profil exact. Ce protocole ne fournit pas le seuil du rapport chronologique.

### Justification

La question imposée par l'organisation des fichiers est celle-ci : que détectent les trois modèles sur un jour dont les attaques n'existaient pas dans l'apprentissage. La répartition 13 est la coupe contiguë qui pose cette question sur le vendredi entier, avec un apprentissage qui contient vraiment la classe Attack.

Les coupes 1 à 6 n'ont aucune attaque à apprendre. Les coupes 7 à 11 n'apprennent que le brute force. Les coupes 12, 14, 16, 17 et 19 fragmentent le jeudi ou le vendredi sans donner une validation assez fournie pour compenser la perte du dernier jour entier. Les coupes 15, 18, 20 et 21 rendent le seuil confortable et laissent au test le seul DDoS. La 21 est la meilleure de ces alternatives : validation PortScan de 158 930 attaques, test DDoS de 128 027 attaques, aucun profil contradictoire entre ces deux ensembles. Elle reste une alternative, parce que l'évaluation officielle ne porterait plus que sur une famille, et que PortScan aurait servi à choisir le seuil.

Le découpage groupé répond à la question complémentaire, la reconnaissance des familles montrées au modèle. Il ne se substitue pas à la question chronologique. Les deux protocoles peuvent être publiés l'un à côté de l'autre, avec des décisions séparées.

Rien de cette recommandation n'est appliqué. Le nettoyage, la constitution des ensembles et l'entraînement attendent la validation.
