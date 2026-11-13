# Rapport du projet

Le système officiel reste le XGBoost entraîné sur les 61 caractéristiques, avec le seuil 0,01 choisi sur le jeudi. L'ablation décrite plus bas est une expérience à part. Elle ne remplace ni le modèle, ni le seuil, ni l'évaluation du vendredi.

Les chiffres de chaque étape viennent du fichier de métriques correspondant. Les rapports de phase gardent l'état du moment où ils ont été écrits.

## Chaîne officielle

Le découpage retenu est chronologique : train du lundi au mercredi, validation le jeudi, test le vendredi. Le découpage groupé 70/15/15 est complémentaire et n'a rien choisi.

| Étape | Résultat officiel | Fichier |
|---|---|---|
| Préparation | 2 520 966 flux conservés, 61 caractéristiques après filtres du train | `docs/preparation_report.md` |

Le nettoyage de ces CSV est justifié plus bas à partir d'Engelen, Rimmer et Joosen [1], sans reprendre leur reconstruction des PCAP.
| Sélection | XGBoost, PR-AUC du jeudi 0,932439, devant la forêt 0,791 et la régression 0,034 | `docs/training_report.md` |
| Seuil | 0,01, coût `FPR + 2 × FNR` sur le jeudi | `docs/threshold_selection.md` |
| Vendredi | PR-AUC 0,806112. Au seuil 0,01 : précision 0,997, rappel 0,382, 259 fausses alertes, 136 015 attaques manquées | `docs/friday_evaluation.md` |
| SHAP | Illustration sur 2 000 flux du vendredi et cinq cas locaux, sans modifier le modèle | `docs/shap_report.md` |
| Audit | Trois variables suspectes : Destination Port, Init_Win_bytes_backward, Init_Win_bytes_forward | `docs/feature_audit.md` |

## Ablation validée

L'expérience exigée retire ces trois variables ensemble, puis réentraîne XGBoost. Les hyperparamètres, la graine 42 et `scale_pos_weight` (6,531) sont ceux du modèle officiel. L'imputation par médiane est ajustée sur le train seulement. Le seuil n'est pas recherché à nouveau. Le vendredi n'est pas scoré. Le fichier `outputs/models/xgboost.joblib` n'est pas modifié. Le modèle d'expérience est `outputs/models/xgboost_without_suspect_features.joblib`.

Sur le jeudi, la PR-AUC passe de **0,932439 à 0,618205**, soit une baisse de 0,314. Cette baisse ne dépend pas du seuil : elle est calculée sur les probabilités continues.

| Modèle | Seuil | Précision | Rappel | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Officiel | 0,50 | 0,953 | 0,832 | 0,889 | 89 | 366 |
| Sans les trois variables | 0,50 | 0,639 | 0,637 | 0,638 | 784 | 792 |
| Officiel | 0,01 | 0,851 | 0,918 | 0,883 | 350 | 178 |
| Sans les trois variables | 0,01 | 0,111 | 0,911 | 0,198 | 15 933 | 193 |

Au seuil 0,50, les fausses alertes et les attaques manquées augmentent toutes les deux. Au seuil 0,01, le rappel reste proche (0,911 contre 0,918), tandis que les fausses alertes passent de 350 à 15 933. Le taux de fausses alertes sur le trafic normal du jeudi passe de 0,088 % à 4,02 %. L'entraînement dure 21,8 s, contre 20,9 s pour le modèle officiel.

Le détail est dans `docs/ablation_suspect_features.md`.

## Limites de cette conclusion

La chute de 0,932 à 0,618 dit que ces trois variables portaient une grande part du classement sur le jeudi. Elle ne dit pas laquelle des trois, parce qu'elles ont été retirées ensemble. Elle ne dit pas non plus ce que le modèle ferait sur le vendredi : Bot, PortScan et DDoS n'ont pas été rescorés.

Le jeudi est une validation de transfert. Ses attaques, surtout des attaques web vers le port 80, sont absentes du train, et leur fenêtre TCP arrière prend une constante rare dans le trafic normal du même jour. La baisse mesure donc le raccourci de scénario sur cette journée. Le seuil 0,01 avait été choisi pour les scores du modèle officiel. Le garder tel quel explique une partie de l'explosion d'alertes. La PR-AUC, elle, baisse indépendamment de ce choix.

La validation a déjà servi à choisir le modèle et le seuil. Ses chiffres ne sont pas une estimation impartiale. Le dernier groupe du parquet déborde de 24 757 lignes sur le vendredi : ces lignes ont été écartées avant la matrice et n'ont pas été scorées. L'ablation ne devient pas le résultat officiel, même si une lecture ultérieure du vendredi lui était favorable.

Engelen, Rimmer et Joosen signalent le même type de raccourci, dans un autre sens [1]. Ils jugent l'usage du port de destination discutable, et leur forêt sépare les artefacts de fin de connexion TCP notamment par `Init Win bytes forward`. Notre ablation ne reprend pas cette expérience. Elle retire trois colonnes que nos propres distributions, sur le train et le jeudi, relient au scénario de collecte. Les constantes de fenêtre que nous mesurons ne sont pas celles qu'ils publient.

## Stabilité des explications

Cette étude est validée. Elle ne remplace pas le modèle officiel. Les fichiers restent ceux de l'expérience : `docs/stability_explanations.md`, `outputs/metrics/stability_explanations.json`, `outputs/tables/stability_top10.csv`, `outputs/tables/stability_rank_agreement.csv`, `outputs/figures/stability_rank_heatmap.png` et `outputs/models/stability/`.

Deux questions ont été séparées.

L'expérience A réentraîne XGBoost sur tout le train officiel, du lundi au mercredi, avec les graines 42, 7 et 123. Les hyperparamètres restent ceux du modèle officiel, dont `subsample=1` et `colsample_bytree=1`. Les trois classements SHAP, calculés sur le même échantillon de 2 000 flux du jeudi, sont identiques : Jaccard du top 10 égal à 1 et Spearman des 61 rangs égal à 1. L'écart maximal entre les moyennes absolues SHAP est nul.

Ce résultat vient des deux paramètres, pas d'un accord entre trois modèles différents. `subsample` est la part des lignes tirée pour chaque arbre. À 1, chaque arbre voit toutes les lignes du train : la graine n'a aucune ligne à choisir. `colsample_bytree` est la part des variables tirée pour chaque arbre. À 1, chaque arbre voit les 61 variables : la graine n'a aucune colonne à choisir. Il ne reste pas d'aléa d'échantillonnage. Avec ces réglages figés, changer 42 en 7 ou en 123 ne change pas l'explication.

L'expérience B ne crée pas un nouveau découpage chronologique. Pour les graines 1001, 1002 et 1003, elle retire à chaque fois 30 % des flux du train officiel, de façon stratifiée sur l'étiquette d'origine. Aucun flux du jeudi ni du vendredi n'entre dans l'apprentissage. L'imputation et `scale_pos_weight` sont ajustés sur le sous-échantillon seul. La graine du modèle reste 42. Le même échantillon du jeudi sert aux six explications.

Le cœur du classement tient. Destination Port est toujours premier, Bwd Packet Length Std toujours deuxième. Les deux fenêtres initiales et Bwd Packet Length Mean restent dans les cinq premiers. Le Jaccard moyen du top 10 est 0,818 et le Spearman moyen des 61 rangs est 0,904. Ce qui bouge est la frontière du top 10 : Fwd IAT Min passe du rang 6, sur le train complet, aux rangs 9, 9 et 12 ; Fwd Header Length entre au rang 10 dans deux sous-échantillons ; Bwd Packet Length Min sort une fois, au rang 11.

## Nettoyage et littérature

La référence exigée est Engelen, Rimmer et Joosen, 2021 [1]. Le texte lu est leur article et la documentation étendue qu'ils publient à part [2]. Ils repartent des PCAP, corrigent CICFlowMeter, reconstruisent les flux et réétiquettent une partie du jeu. Plus de 20 % des traces d'origine sont reconstruites ou réétiquetées. Nous n'avons pas refait ce travail. Les CSV publics restent la source, lus sans être modifiés. Le journal de nos opérations est `docs/cleaning_log.md`.

Leur article décrit quatre familles de défauts. La construction des flux termine une connexion TCP au premier drapeau FIN, contrairement à la spécification. Les paquets qui suivent forment un flux séparé, qu'ils appellent appendice TCP. Ces appendices représentent 25,9 % du jeu, et plus de 25 % des flux sont, selon eux, des artefacts sans caractère identifiable. Un appendice qui expire peut inverser le sens du flux suivant. Comme l'étiquette dépend du sens, une attaque peut alors être marquée bénigne. L'outil ne traite pas non plus le drapeau RST comme une fin de connexion. L'étiquetage, ensuite, ne vérifie pas le contenu : il utilise l'adresse de l'attaquant, celle de la victime et une fenêtre de temps. Des flux sans charge utile aller, notamment pour les attaques web, le botnet et certains dénis de service, sont donc étiquetés comme des attaques alors que l'action malveillante n'y est pas portée. Ils proposent de les marquer « Attempted ». L'outil DoS Hulk utilisé pour la capture envoie `Connection: Close` au lieu de `Keep-Alive`, ce qui rend cette attaque peu représentative. Enfin, ils montrent qu'une forêt mémorise des raccourcis : longueur d'en-tête aller, nombre de paquets aller, taille minimale de segment et fenêtre initiale aller pour séparer les appendices, et le port de destination reste, selon leurs mots, un attribut dont l'usage est discutable.

La documentation étendue ajoute un défaut que l'article ne détaille pas. La colonne `Fwd Header Length` est dupliquée dans les CSV, à cause d'une erreur de CICFlowMeter déjà corrigée dans l'outil mais encore présente dans le jeu publié [2]. Elle signale aussi que les compteurs de drapeaux TCP ne valent que 0 ou 1, parce que seul le premier paquet était compté. Elle précise qu'une autre erreur, l'horodatage absolu dans les durées d'activité et d'inactivité, n'affectait pas CICIDS2017 au moment de sa création.

Le tableau suivant relie ces observations à nos opérations. Une ligne n'est marquée confirmée que si les auteurs décrivent le même défaut. Les effectifs sont les nôtres, mesurés sur les huit CSV.

| Opération réalisée | Effectif mesuré ici | Lien avec Engelen et al. |
|---|---|---|
| Seconde colonne `Fwd Header Length` retirée, après égalité constatée sur les 2 830 743 lignes | 1 colonne en double | Confirmé par la documentation étendue [2]. Notre contrôle d'égalité est indépendant. Nous n'avons pas régénéré les flux. |
| Copies exactes : première occurrence chronologique conservée | Inventaire brut : 308 381 copies supplémentaires, soit 2 830 743 − 2 522 362. Après exclusion des 7 020 lignes contradictoires : 302 757 copies retirées, 2 520 966 lignes gardées. L'écart, 5 624, est le nombre de copies qui se trouvaient dans ces groupes : 7 020 − 1 396 | Non décrit par les auteurs. Le retrait évite qu'un même contenu figure dans l'apprentissage et dans le test. Leurs appendices TCP sont un autre objet : des flux supplémentaires produits par la coupure au premier FIN, que nous n'avons pas reconstruits. |
| Infinis positifs remplacés par une valeur manquante, lignes conservées | Inventaire brut : 4 376 cellules, dont 1 509 dans `Flow Bytes/s` et 2 867 dans `Flow Packets/s`. Sur les lignes écrites dans le fichier dérivé : 2 773 cellules remplacées, puis 0 infini | Non chiffré dans l'article ni dans la documentation étendue consultées. L'opération vient de notre inspection. Elle empêche un débit infini, produit par une durée nulle, d'entrer dans les modèles. |
| Jetons manquants conservés, sans imputation à ce stade | Inventaire brut : 1 358 jetons `NaN`, tous dans `Flow Bytes/s`. Sur les lignes conservées : 351. Avec les 2 773 infinis remplacés, 3 124 cellules manquantes, 1 562 dans chacune des deux colonnes de débit | Non chiffré par les auteurs. L'imputation médiane est faite plus tard, sur le train seulement. |
| Profils aux variables identiques et aux étiquettes différentes, groupe entier exclu | 698 profils, 7 020 lignes | Non comptés par les auteurs. Leur critique de l'étiquetage porte sur le sens du flux et sur les connexions sans charge utile, pas sur ces groupes. L'exclusion évite d'apprendre le même profil avec deux classes. |

Les autres défauts qu'ils établissent restent donc ouverts dans notre fichier dérivé. Les appendices TCP n'ont pas été recollés. Les flux sans charge utile n'ont pas reçu l'étiquette « Attempted ». DoS Hulk n'a pas été retiré. Le port de destination est resté dans le modèle officiel. L'ablation qui le retire, avec les deux fenêtres initiales, est une expérience séparée. Elle ne réalise pas leur reconstruction du jeu.

## Hors périmètre

Les partitions 70/15/15 d'origine, pour les graines 1001, 1002 et 1003, n'ont pas été créées. Elles auraient mélangé le vendredi dans l'apprentissage. L'expérience B validée les remplace. Le modèle d'ablation n'est pas comparé sur le vendredi. Les trois variables suspectes ne sont pas retirées une par une. Aucune de ces suites n'est lancée.

## Bibliographie

[1] Engelen, G., Rimmer, V. et Joosen, W. (2021). Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study. *2021 IEEE Security and Privacy Workshops (SPW)*, p. 7-12. DOI : [10.1109/SPW53761.2021.00009](https://doi.org/10.1109/SPW53761.2021.00009). Texte consulté : [IEEE Xplore](https://ieeexplore.ieee.org/document/9474286) et le PDF déposé par les auteurs, [wtmc2021_Engelen_Troubleshooting.pdf](https://intrusion-detection.distrinet-research.be/WTMC2021/Resources/wtmc2021_Engelen_Troubleshooting.pdf).

[2] Engelen, G., Rimmer, V. et Joosen, W. Documentation étendue du même article. Consultée à l'adresse [https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html](https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html). La page d'accueil du complément est [https://intrusion-detection.distrinet-research.be/WTMC2021/](https://intrusion-detection.distrinet-research.be/WTMC2021/).
