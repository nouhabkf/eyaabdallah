# Journal de nettoyage

Préparation du 2026-11-13T00:22:01.804006+00:00. Les huit CSV bruts n'ont pas été modifiés. Leurs empreintes SHA-256 sont dans `outputs/metrics/preparation_manifest.json`. Aucun modèle n'a été ajusté.

| Étape | Règle | Avant | Après | Décision |
| --- | --- | ---: | ---: | --- |
| Lecture | Huit CSV bruts, ouverts en lecture seule | 2830743 | 2830743 | Aucune écriture dans `raw` |
| Noms | Espaces de début et de fin retirés | 79 colonnes | 79 noms nets | Le nom brut n'est pas réécrit |
| Fwd Header Length | Les deux colonnes sont égales sur toutes les lignes | 2 colonnes | 1 colonne | La seconde occurrence est absente du fichier dérivé |
| Infinis | Remplacés par une valeur manquante | 2773 cellules | 0 infini dans le parquet | Les lignes restent |
| Manquants | Jetons vides ou `NaN` conservés comme manquants | 351 cellules | 3124 cellules manquantes | Pas d'imputation |
| Profils contradictoires | 698 profils, colonnes CSV identiques, étiquettes différentes | 7020 lignes | 0 profil restant | Groupe entier exclu |
| Doublons exacts | Première occurrence chronologique de la ligne brute | 302757 copies | 2520966 lignes | 28011 contenus non contradictoires empêchés de traverser deux ensembles ; 13 autres étaient déjà exclus avec leur profil contradictoire |

## Effectifs chronologiques

| Ensemble | Lignes | Normal | Attack |
| --- | ---: | ---: | ---: |
| train | 1527133 | 1324353 | 202780 |
| validation | 398110 | 395931 | 2179 |
| test | 595723 | 375502 | 220221 |

Le fichier dérivé est `C:/data/cic-ids2017/interim/flows.parquet`. Les indices sont sous `C:/data/cic-ids2017/splits/`.
