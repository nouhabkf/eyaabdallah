# Rapport d'inspection — CIC-IDS2017

Contrôle du 13 novembre 2026. Lecture seule des huit CSV placés dans `C:\data\cic-ids2017\raw`.

Aucun fichier brut n'a été modifié. Aucun modèle n'a été ajusté. SHAP n'a pas été calculé. Le nettoyage définitif n'a pas commencé. Le découpage définitif n'est pas choisi.

Les trois algorithmes restent prévus : régression logistique, forêt aléatoire et XGBoost.

Les chiffres ci-dessous viennent de `scripts/inspect_raw.py`. Le détail machine est dans `outputs/metrics/inspection.json`. Pour chaque fichier, le nombre de lignes lu octet par octet est identique au nombre de lignes lu par pandas.

Espace disque au moment de la vérification : 26,23 Go libres sur le disque C.

## Fichiers

Huit CSV, 884 645 759 octets au total. Encodage UTF-8 sans erreur de décodage, fins de ligne LF, aucune ligne vide. Les huit en-têtes sont identiques : 79 colonnes.

| Fichier | Jour indiqué par le nom | Lignes | Octets |
| --- | --- | ---: | ---: |
| `Monday-WorkingHours.pcap_ISCX.csv` | Lundi | 529 918 | 176 927 918 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Mardi | 445 909 | 135 078 995 |
| `Wednesday-workingHours.pcap_ISCX.csv` | Mercredi | 692 703 | 225 166 395 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Jeudi matin | 170 366 | 52 023 263 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Jeudi après-midi | 288 602 | 83 102 436 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Vendredi matin | 191 033 | 58 316 725 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Vendredi après-midi, PortScan | 286 467 | 76 906 168 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Vendredi après-midi, DDoS | 225 745 | 77 123 859 |
| **Total** |  | **2 830 743** | **884 645 759** |

Le nom du fichier du jeudi après-midi contient l'orthographe `Infilteration`. L'étiquette correspondante dans les données est `Infiltration`.

## Colonnes et types

Les noms sont reproduits tels qu'ils sont dans l'en-tête, espaces compris. Sur les morceaux lus, 54 colonnes sont de type entier, 24 de type flottant, et ` Label` est du texte. Ce typage est le même dans les huit fichiers.

Colonnes flottantes : ` Fwd Packet Length Mean`, ` Fwd Packet Length Std`, ` Bwd Packet Length Mean`, ` Bwd Packet Length Std`, `Flow Bytes/s`, ` Flow Packets/s`, ` Flow IAT Mean`, ` Flow IAT Std`, ` Fwd IAT Mean`, ` Fwd IAT Std`, ` Bwd IAT Mean`, ` Bwd IAT Std`, `Fwd Packets/s`, ` Bwd Packets/s`, ` Packet Length Mean`, ` Packet Length Std`, ` Packet Length Variance`, ` Average Packet Size`, ` Avg Fwd Segment Size`, ` Avg Bwd Segment Size`, `Active Mean`, ` Active Std`, `Idle Mean`, ` Idle Std`.

Deux problèmes de noms sont constants :

- La plupart des noms commencent par un espace. Quelques-uns n'en ont pas, dont `Flow Bytes/s`, `Total Length of Fwd Packets` et `Label` précédé d'un espace.
- Le nom ` Fwd Header Length` est présent deux fois, en positions 35 et 56. Pendant la lecture en mémoire, la seconde occurrence a été distinguée sous le nom ` Fwd Header Length.1`. Le fichier brut, lui, conserve les deux noms identiques.

## Dates et heures

Aucune colonne de date ni d'heure n'existe. `Flow Duration`, les colonnes IAT, `Active` et `Idle` sont des durées numériques, pas une horloge.

Le jour n'est connu que par le nom du fichier. Un découpage à l'heure, à l'intérieur d'une journée, n'est pas possible avec ces CSV seuls.

## Étiquettes

La colonne d'étiquette ne contient aucune valeur manquante. Le comptage binaire ci-dessous ne modifie pas les étiquettes : `BENIGN` est compté comme Normal, toute autre étiquette comme Attack.

| Étiquette lue | Lignes |
| --- | ---: |
| BENIGN | 2 273 097 |
| DoS Hulk | 231 073 |
| PortScan | 158 930 |
| DDoS | 128 027 |
| DoS GoldenEye | 10 293 |
| FTP-Patator | 7 938 |
| SSH-Patator | 5 897 |
| DoS slowloris | 5 796 |
| DoS Slowhttptest | 5 499 |
| Bot | 1 966 |
| Web Attack, puis le caractère U+FFFD, puis Brute Force | 1 507 |
| Web Attack, puis U+FFFD, puis XSS | 652 |
| Infiltration | 36 |
| Web Attack, puis U+FFFD, puis Sql Injection | 21 |
| Heartbleed | 11 |

Normal : 2 273 097 lignes, soit 80,34 % du total. Attack : 557 646 lignes, soit 19,66 %.

Les trois étiquettes d'attaque web contiennent, dans le fichier brut, les octets UTF-8 `EF BF BD`. C'est le caractère de remplacement Unicode U+FFFD, pas un tiret. Il se trouve entre `Web Attack` et le type d'attaque. Cette forme a été lue, pas corrigée.

## Répartition par fichier

| Fichier | Normal | Attack | Part d'attaques | Familles d'attaque |
| --- | ---: | ---: | ---: | --- |
| Lundi | 529 918 | 0 | 0 % | aucune |
| Mardi | 432 074 | 13 835 | 3,10 % | FTP-Patator 7 938 ; SSH-Patator 5 897 |
| Mercredi | 440 031 | 252 672 | 36,48 % | DoS Hulk 231 073 ; DoS GoldenEye 10 293 ; DoS slowloris 5 796 ; DoS Slowhttptest 5 499 ; Heartbleed 11 |
| Jeudi matin | 168 186 | 2 180 | 1,28 % | Web Attack Brute Force 1 507 ; XSS 652 ; Sql Injection 21 |
| Jeudi après-midi | 288 566 | 36 | 0,012 % | Infiltration 36 |
| Vendredi matin | 189 067 | 1 966 | 1,03 % | Bot 1 966 |
| Vendredi PortScan | 127 537 | 158 930 | 55,48 % | PortScan 158 930 |
| Vendredi DDoS | 97 718 | 128 027 | 56,71 % | DDoS 128 027 |

Chaque famille d'attaque n'apparaît que dans un seul fichier.

## Valeurs manquantes et infinies

Les seules valeurs manquantes sont dans `Flow Bytes/s` : 1 358 cellules.

Les seules valeurs infinies sont positives :

| Colonne | Infinis positifs |
| --- | ---: |
| `Flow Bytes/s` | 1 509 |
| ` Flow Packets/s` | 2 867 |
| **Total** | **4 376** |

Aucun infini négatif n'a été trouvé. Les autres colonnes n'ont ni valeur manquante ni valeur infinie sur l'ensemble des 2 830 743 lignes.

## Doublons exacts

Une ligne est identique à une autre lorsque l'empreinte MD5 de son contenu brut, sans le saut de ligne, est la même.

| Mesure | Valeur |
| --- | ---: |
| Lignes | 2 830 743 |
| Contenus distincts | 2 522 362 |
| Copies supplémentaires dans l'ensemble des fichiers | 308 381 |
| Copies supplémentaires à l'intérieur d'un même fichier | 256 479 |
| Contenus présents dans plusieurs fichiers | 33 563 |
| Lignes portées par ces contenus partagés entre fichiers | 139 843 |

Copies supplémentaires à l'intérieur de chaque fichier : PortScan 72 353, mercredi 81 909, jeudi après-midi 35 630, lundi 26 935, mardi 24 065, vendredi matin 6 888, jeudi matin 6 066, DDoS 2 633.

## Conséquences pour le futur découpage

Le découpage n'est pas choisi. Trois faits devront être pris en compte avant de le figer :

- Sans colonne d'horloge, le seul ordre disponible est celui des fichiers, donc des journées et des demi-journées.
- Une séparation par jour place des familles d'attaque entières d'un seul côté. Le lundi ne contient aucune attaque. Le jeudi après-midi n'en contient que 36.
- Un découpage aléatoire mélange les mêmes familles des deux côtés. Il expose aussi au partage des 308 381 copies supplémentaires et de flux issus de la même campagne.

Le choix reste en attente de validation.
