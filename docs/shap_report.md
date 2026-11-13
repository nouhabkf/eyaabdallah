# Explications SHAP du XGBoost sur le vendredi

Le modèle enregistré est relu. Il n'est pas réentraîné. Le seuil officiel reste 0,01, déjà choisi sur le jeudi. Les métriques du vendredi déjà publiées ne sont pas recalculées comme un nouveau résultat et leurs fichiers ne sont pas réécrits. Aucune variable n'est retirée. L'audit et l'analyse de stabilité ne sont pas commencés.

## Méthode

L'explicateur est `shap.TreeExplainer` 0.54.1, sur CPU, avec `feature_perturbation=tree_path_dependent` et `model_output=raw`. KernelSHAP n'est pas utilisé. Aucun jeu de fond n'est tiré : les effectifs déjà enregistrés dans les arbres servent de référence. La classe expliquée est l'attaque, codée 1. L'échelle est le log-odds de cette classe, la marge brute du modèle, pas la probabilité. Une valeur SHAP positive augmente le log-odds de l'attaque. La probabilité s'obtient par la fonction sigmoïde de la marge reconstruite.

Les scores utilisés pour classer les flux sont ceux de `predict_proba` de la classe attaque, au seuil 0,01. Cette relecture reproduit la matrice de confusion publiée : TN 375243, FP 259, FN 136015, TP 84206. La PR-AUC relue est 0.806112.

L'échantillon global compte 2000 flux, tirés avec la graine 42 de façon proportionnelle aux étiquettes d'origine du vendredi. Le plafond appliqué est 2000. Ce tirage illustre le test. Il ne choisit ni une variable, ni un modèle, ni un seuil. Le SHAP de validation prévu pour l'audit n'est pas calculé.

Les cinq cas suivent la règle `scores_and_labels_only_v1`, appliquée à tout le vendredi avant le calcul des graphiques locaux. En cas d'égalité de score, le plus petit indice du vendredi est gardé. Une catégorie vide resterait absente : aucun exemple de remplacement n'est créé.

Cohérence : la somme de la valeur de base et des contributions reconstruit la marge du modèle avec un écart maximal de 7.629e-06. L'écart maximal entre la probabilité reconstruite et `predict_proba` est 7.749e-07. Les décisions au seuil 0,01 coïncident sur les 2005 flux vérifiés.

Temps : construction de l'explicateur 0.014 s, explication des 2 000 flux 0.107 s, explication des cas 0.005 s.

## Importance globale

Le classement est la moyenne des valeurs absolues SHAP sur les 2 000 flux. La moyenne signée est donnée à côté : elle dit le sens moyen, pas le rang.

| Rang | Variable | Moyenne absolue | Moyenne signée |
|---:|---|---:|---:|
| 1 | Destination Port | 3.597745 | -3.052439 |
| 2 | Bwd Packet Length Std | 2.742078 | -0.420192 |
| 3 | Init_Win_bytes_backward | 2.574080 | -2.541502 |
| 4 | Bwd Packet Length Mean | 1.474278 | -1.196251 |
| 5 | Init_Win_bytes_forward | 1.171452 | -1.048709 |
| 6 | Bwd Packets/s | 0.601495 | -0.565020 |
| 7 | Fwd IAT Min | 0.597863 | -0.362038 |
| 8 | Bwd Packet Length Min | 0.528766 | -0.489263 |
| 9 | min_seg_size_forward | 0.487485 | -0.421219 |
| 10 | Bwd Packet Length Max | 0.435125 | -0.255263 |

Répartition de l'échantillon :

| Étiquette | Vendredi | Échantillon |
|---|---:|---:|
| BENIGN | 375502 | 1261 |
| Bot | 1953 | 6 |
| DDoS | 128013 | 430 |
| PortScan | 90255 | 303 |

## Cas individuels

Les cinq commentaires ci-dessous lisent les valeurs déjà enregistrées. Aucun SHAP n'a été recalculé. La valeur de base est la même pour les cinq flux, −0,025 en log-odds, soit une probabilité proche de 0,5 avant de regarder les variables. Le seuil 0,01 correspond à un log-odds d'environ −4,595 : au-dessus, le modèle annonce une attaque. Une contribution positive pousse vers l'attaque, une contribution négative vers le trafic normal. Les graphiques waterfall déjà produits sont conservés.

### tp_confident

Indice vendredi 352055, ligne 2277298 du parquet. Étiquette d'origine : PortScan. Vraie classe : attaque. Score 0.99997044. Au seuil 0,01, la prédiction est attaque. 1 flux partage ce score parmi 84206 éligibles ; le plus petit indice est retenu.

| Rang | Variable | Valeur vue par le modèle | SHAP (log-odds) | Sens |
|---:|---|---:|---:|---|
| 1 | Destination Port | 80.000000 | 3.213365 | vers l'attaque |
| 2 | Bwd Packet Length Mean | 0.000000 | 2.740691 | vers l'attaque |
| 3 | Fwd Header Length | 160.000000 | 2.730068 | vers l'attaque |
| 4 | Init_Win_bytes_backward | -1.000000 | -1.551489 | vers le normal |
| 5 | Init_Win_bytes_forward | 251.000000 | 1.539003 | vers l'attaque |
| 6 | Fwd IAT Min | 3.000000 | 0.884766 | vers l'attaque |
| 7 | Bwd Packet Length Max | 0.000000 | 0.550904 | vers l'attaque |
| 8 | Packet Length Std | 0.000000 | -0.250492 | vers le normal |
| 9 | Flow IAT Mean | 599.000000 | 0.228931 | vers l'attaque |
| 10 | Flow IAT Std | 724.009644 | 0.228161 | vers l'attaque |

Ce flux est une attaque, de famille PortScan. Le modèle annonce aussi une attaque. Le score vaut 0,99997044 et la marge reconstruite 10,427. C'est le vrai positif le plus sûr du vendredi.

Les variables qui pèsent le plus sont le port de destination 80 (+3,213), une longueur moyenne de paquets retour nulle (+2,741), une longueur d'en-tête aller de 160 octets (+2,730) et la fenêtre initiale aller, égale à 251 (+1,539). Le plus petit intervalle entre paquets aller vaut 3 microsecondes et pousse encore vers l'attaque (+0,885). La longueur maximale retour, nulle elle aussi, ajoute +0,551.

Deux contributions vont dans l'autre sens. La fenêtre initiale retour vaut −1, le code d'absence déjà observé sur une grande part du trafic normal du train, et elle retire 1,551. L'écart-type des longueurs de paquets, nul, retire 0,250. Les intervalles moyens et dispersés du flux, 599 et 724 microsecondes, ajoutent chacun environ +0,229.

La décision est nette parce que le port 80, l'absence de réponse et l'en-tête aller l'emportent largement sur la fenêtre manquante. Le modèle ne voit pas un balayage de nombreux ports : ce flux-ci vise le port 80, et c'est cette valeur, jointe à une réponse vide et à un intervalle très court, qui produit la quasi-certitude.

Commentaire d'analyste. Alerte à traiter en priorité. Le flux est étiqueté PortScan et le score est saturé. Le motif retenu est un flux vers le port 80, sans charge utile au retour, avec un en-tête aller de 160 octets et un écart de 3 microsecondes. La fenêtre retour absente freine l'alerte, sans la renverser. Ce motif n'est pas une signature propre au balayage : le faux positif sûr, plus bas, réunit le port 80 et une réponse vide. Ici, l'étiquette confirme l'attaque. L'explication dit surtout que le détecteur a reconnu un flux court et sans réponse vers un service web.

Graphique : `outputs/figures/shap_waterfall_tp_confident.png`.

### tp_marginal

Indice vendredi 532774, ligne 2458017 du parquet. Étiquette d'origine : DDoS. Vraie classe : attaque. Score 0.01000577. Au seuil 0,01, la prédiction est attaque. 1 flux partage ce score parmi 84206 éligibles ; le plus petit indice est retenu.

| Rang | Variable | Valeur vue par le modèle | SHAP (log-odds) | Sens |
|---:|---|---:|---:|---|
| 1 | Init_Win_bytes_forward | 256.000000 | -3.148155 | vers le normal |
| 2 | Destination Port | 80.000000 | 2.103768 | vers l'attaque |
| 3 | Init_Win_bytes_backward | -1.000000 | -1.733173 | vers le normal |
| 4 | Bwd Packet Length Mean | 0.000000 | 1.602297 | vers l'attaque |
| 5 | Fwd IAT Min | 988.000000 | 1.241656 | vers l'attaque |
| 6 | min_seg_size_forward | 20.000000 | -1.218892 | vers le normal |
| 7 | Fwd Header Length | 120.000000 | 1.072822 | vers l'attaque |
| 8 | Bwd Packet Length Std | 0.000000 | -0.993418 | vers le normal |
| 9 | Idle Min | 8267501.000000 | 0.629819 | vers l'attaque |
| 10 | Fwd Packet Length Min | 6.000000 | -0.606922 | vers le normal |

Ce flux est une attaque, de famille DDoS. Le modèle annonce aussi une attaque, mais de justesse. Le score vaut 0,01000577, donc juste au-dessus de 0,01. La marge reconstruite vaut −4,595.

La plus forte contribution est négative : la fenêtre initiale aller vaut 256 et retire 3,148. La fenêtre retour, absente (−1), retire encore 1,733. La taille minimale de segment aller, 20, retire 1,219. L'écart-type des longueurs retour, nul, retire 0,993, et la plus petite longueur aller, 6 octets, retire 0,607.

Du côté de l'attaque, le port 80 ajoute 2,104, la longueur moyenne retour nulle ajoute 1,602, un intervalle minimal aller de 988 microsecondes ajoute 1,242, et l'en-tête aller de 120 octets ajoute 1,073. Le plus petit temps d'inactivité, 8 267 501 microsecondes, soit environ 8,3 secondes, ajoute 0,630.

Ces deux groupes se compensent. La marge reste négative, proche de la frontière du seuil. Le port, la réponse vide et la pause d'inactivité suffisent tout juste à passer 0,01. Les deux fenêtres, surtout celle de 256 octets, retiennent le score au bord de l'alerte.

Commentaire d'analyste. Vrai positif fragile. L'étiquette est DDoS, mais le score ne dépasse le seuil que de quelques millionièmes. Je ne traiterais pas cette alerte comme un flux volumineux confirmé : le retour est vide, et la décision est une somme presque nulle. Le port 80 et l'absence de réponse tirent vers l'attaque. La fenêtre aller de 256 octets tire vers le trafic ordinaire. En file d'attente, ce flux irait en revue manuelle, avec la famille DDoS comme contexte, et non comme une alerte déjà établie par le score.

Graphique : `outputs/figures/shap_waterfall_tp_marginal.png`.

### fp_confident

Indice vendredi 181994, ligne 2107237 du parquet. Étiquette d'origine : BENIGN. Vraie classe : normal. Score 0.99961889. Au seuil 0,01, la prédiction est attaque. 5 flux partagent ce score parmi 259 éligibles ; le plus petit indice est retenu.

| Rang | Variable | Valeur vue par le modèle | SHAP (log-odds) | Sens |
|---:|---|---:|---:|---|
| 1 | Destination Port | 80.000000 | 2.988565 | vers l'attaque |
| 2 | Bwd Packet Length Mean | 0.000000 | 2.637384 | vers l'attaque |
| 3 | Init_Win_bytes_backward | -1.000000 | -1.532981 | vers le normal |
| 4 | Fwd Header Length | 280.000000 | 1.465361 | vers l'attaque |
| 5 | Flow IAT Std | 11900000.000000 | 1.171826 | vers l'attaque |
| 6 | Active Min | 7013505.000000 | -0.590811 | vers le normal |
| 7 | Bwd Packet Length Max | 0.000000 | 0.546883 | vers l'attaque |
| 8 | Flow IAT Min | 997510.000000 | 0.422273 | vers l'attaque |
| 9 | Idle Mean | 18700000.000000 | 0.386318 | vers l'attaque |
| 10 | Idle Min | 8016015.000000 | 0.267691 | vers l'attaque |

Ce flux est du trafic normal, étiquette BENIGN. Ce n'est pas une attaque. Le modèle annonce pourtant une attaque, avec un score de 0,99961889 et une marge reconstruite de 7,872. Cinq flux normaux partagent ce score. Le cas retenu est le premier dans l'ordre du vendredi.

Le port 80 ajoute 2,989 et la longueur moyenne retour, nulle, ajoute 2,637. L'en-tête aller de 280 octets ajoute 1,465. L'écart-type des intervalles du flux, 11 900 000 microsecondes, soit environ 11,9 secondes, ajoute 1,172. La longueur maximale retour, nulle, ajoute 0,547. Le plus petit intervalle du flux, 997 510 microsecondes, ajoute 0,422. Les temps d'inactivité, en moyenne 18 700 000 microsecondes et au minimum 8 016 015, ajoutent 0,386 et 0,268.

La fenêtre retour absente retire 1,533. Le plus petit temps d'activité, 7 013 505 microsecondes, retire 0,591. Ces deux freins ne compensent pas le port, la réponse vide et les longues pauses.

La décision confiante vient donc d'un flux web sans charge utile au retour, entrecoupé de silences de plusieurs secondes. Le modèle a appris à lire cette forme comme une attaque. Ici, l'étiquette dit que le flux est légitime. La certitude du score ne corrige pas cette erreur.

Commentaire d'analyste. Fausse alerte, à ne pas confondre avec le PortScan précédent. Le score est presque aussi haut, et le port 80 plus une réponse vide reviennent. La différence mesurée est le rythme : ici les silences durent plusieurs secondes, alors que le vrai positif sûr avait un écart aller de 3 microsecondes. Un silence long sur le port 80 peut être une connexion HTTP inactive. L'étiquette BENIGN le confirme. Je classerais l'alerte comme un faux positif dû au raccourci « service web sans réponse », et non comme un incident. Cinq flux normaux portent exactement ce score : le cas montré est le premier du fichier, pas un exemple choisi pour son récit.

Graphique : `outputs/figures/shap_waterfall_fp_confident.png`.

### fp_marginal

Indice vendredi 182411, ligne 2107654 du parquet. Étiquette d'origine : BENIGN. Vraie classe : normal. Score 0.01012878. Au seuil 0,01, la prédiction est attaque. 2 flux partagent ce score parmi 259 éligibles ; le plus petit indice est retenu.

| Rang | Variable | Valeur vue par le modèle | SHAP (log-odds) | Sens |
|---:|---|---:|---:|---|
| 1 | Init_Win_bytes_forward | 264.000000 | -3.123647 | vers le normal |
| 2 | Destination Port | 80.000000 | 1.931628 | vers l'attaque |
| 3 | Init_Win_bytes_backward | -1.000000 | -1.705076 | vers le normal |
| 4 | Bwd Packet Length Mean | 0.000000 | 1.542980 | vers l'attaque |
| 5 | Bwd Packet Length Std | 0.000000 | -1.078590 | vers le normal |
| 6 | Flow IAT Std | 0.000000 | -0.969391 | vers le normal |
| 7 | Fwd IAT Max | 141.000000 | -0.566378 | vers le normal |
| 8 | Fwd IAT Min | 141.000000 | -0.318688 | vers le normal |
| 9 | Bwd Packet Length Max | 0.000000 | 0.315898 | vers l'attaque |
| 10 | Flow IAT Mean | 141.000000 | 0.315022 | vers l'attaque |

Ce flux est du trafic normal, étiquette BENIGN. Ce n'est pas une attaque. Le modèle l'annonce pourtant comme une attaque. Le score vaut 0,01012878, juste au-dessus de 0,01, et la marge reconstruite vaut −4,582. Deux flux normaux partagent ce score. Le plus petit indice est gardé.

La fenêtre aller vaut 264 et retire 3,124, la plus forte contribution. La fenêtre retour absente retire 1,705. L'écart-type des longueurs retour, nul, retire 1,079. L'écart-type des intervalles du flux, nul, retire 0,969. Les intervalles aller, maximum et minimum égaux à 141 microsecondes, retirent 0,566 et 0,319.

Le port 80 ajoute 1,932 et la longueur moyenne retour, nulle, ajoute 1,543. La longueur maximale retour, nulle, ajoute 0,316. L'intervalle moyen du flux, 141 microsecondes, ajoute 0,315.

Le flux est régulier, à 141 microsecondes, et sa fenêtre aller pousse fortement vers le trafic normal. Le port 80 et l'absence de réponse suffisent néanmoins à franchir le seuil. L'erreur est donc un dépassement de très peu, produit par le même couple port et réponse vide que dans les autres cas, malgré un rythme régulier que le modèle lit plutôt comme normal.

Commentaire d'analyste. Fausse alerte de bord. Le trafic est normal, et le score ne dépasse 0,01 que de très peu. Je n'ouvrirais pas d'incident. Le port 80 et une réponse de longueur nulle tirent vers l'attaque. La fenêtre de 264 octets et des intervalles tous égaux à 141 microsecondes tirent vers une connexion courte et régulière. Cette régularité est compatible avec un échange bref, pas avec une preuve d'attaque. Le classement en alerte vient du seuil bas, pas d'une conviction du modèle : la marge reste très négative.

Graphique : `outputs/figures/shap_waterfall_fp_marginal.png`.

### fn_marginal

Indice vendredi 442290, ligne 2367533 du parquet. Étiquette d'origine : DDoS. Vraie classe : attaque. Score 0.00999992. Au seuil 0,01, la prédiction est normal. 2 flux partagent ce score parmi 136015 éligibles ; le plus petit indice est retenu.

| Rang | Variable | Valeur vue par le modèle | SHAP (log-odds) | Sens |
|---:|---|---:|---:|---|
| 1 | Bwd Packet Length Std | 4128.319336 | 8.426524 | vers l'attaque |
| 2 | Init_Win_bytes_backward | 229.000000 | -7.770087 | vers le normal |
| 3 | Init_Win_bytes_forward | 8192.000000 | -1.766950 | vers le normal |
| 4 | Bwd Packet Length Mean | 2900.250000 | -0.931990 | vers le normal |
| 5 | Fwd IAT Min | 58.000000 | -0.908874 | vers le normal |
| 6 | PSH Flag Count | 1.000000 | 0.737631 | vers l'attaque |
| 7 | Bwd IAT Min | 178.000000 | -0.568805 | vers le normal |
| 8 | Bwd Header Length | 92.000000 | -0.564978 | vers le normal |
| 9 | min_seg_size_forward | 20.000000 | -0.399523 | vers le normal |
| 10 | Average Packet Size | 1661.000000 | 0.398728 | vers l'attaque |

Ce flux est une attaque, de famille DDoS. Le modèle annonce du trafic normal. Le score vaut 0,00999992, donc juste sous 0,01. La marge reconstruite vaut −4,595. Deux attaques partagent ce score. Le plus petit indice est gardé. C'est un faux négatif : l'attaque est manquée.

La dispersion des longueurs retour vaut 4 128,32 et pousse très fort vers l'attaque, +8,427. Le drapeau PSH, présent, ajoute 0,738. La taille moyenne de paquet, 1 661 octets, ajoute 0,399.

La fenêtre retour vaut 229 et retire 7,770, presque autant que la dispersion n'avait ajouté. La fenêtre aller, 8 192, retire 1,767. La longueur moyenne retour, 2 900,25 octets, retire 0,932 : une réponse fournie tire ici vers le normal, alors qu'une réponse vide tirait vers l'attaque dans les quatre autres cas. L'intervalle minimal aller, 58 microsecondes, retire 0,909. L'intervalle minimal retour, 178 microsecondes, retire 0,569. La longueur d'en-tête retour, 92 octets, retire 0,565. La taille minimale de segment, 20, retire 0,400.

L'erreur vient de cette compensation. Le modèle voit bien une réponse longue et variable, et il la compte comme un indice d'attaque. La valeur 229 de la fenêtre retour annule presque tout ce crédit. Avec la fenêtre aller de 8 192, la somme passe juste sous le seuil. Le flux n'a pas la forme des alertes précédentes, qui reposaient sur une réponse vide vers le port 80. Il ressemble à un échange déjà constitué, et la fenêtre initiale suffit à le faire classer comme normal.

Commentaire d'analyste. Attaque manquée. L'étiquette est DDoS et aucune alerte n'est levée. Je ne lirais pas ce silence comme une absence de signal : l'écart-type des longueurs retour, 4 128 octets, contribue à lui seul +8,427 vers l'attaque, et le drapeau PSH va dans le même sens. La décision bascule à cause de la fenêtre retour, égale à 229, dont la contribution est −7,770. Le détecteur rate donc un flux qui transporte une réponse volumineuse, parce qu'une seule valeur de fenêtre, apprise comme un marqueur de trafic ordinaire, couvre le reste. En investigation, ce faux négatif montre la limite du seuil 0,01 autant que celle du motif : le score est à 0,00000008 du seuil, et le motif qui gagne n'est pas le volume échangé.

Graphique : `outputs/figures/shap_waterfall_fn_marginal.png`.

## Limites

Les contributions se partagent le long des chemins vus à l'entraînement. Des variables corrélées se répartissent le crédit : une forte valeur SHAP ne prouve pas qu'une variable agirait seule. L'échantillon de 2 000 flux décrit le vendredi, il ne le remplace pas. Les familles du vendredi, Bot, PortScan et DDoS, sont absentes de l'entraînement. Lorsque plusieurs flux ont exactement le même score extrême, le cas retenu est le premier dans l'ordre du fichier, pas un flux choisi pour son intérêt visuel.

Ces explications ne modifient pas le modèle, le seuil, ni les métriques déjà publiées.

## Fichiers

- `outputs/metrics/shap_explanation.json`
- `outputs/tables/shap_global_importance.csv`
- `outputs/tables/shap_cases.csv`
- `outputs/tables/shap_case_contributions.csv`
- `outputs/figures/shap_summary_beeswarm.png`
- `outputs/figures/shap_bar_top10.png`
- `outputs/figures/shap_waterfall_tp_confident.png`
- `outputs/figures/shap_waterfall_tp_marginal.png`
- `outputs/figures/shap_waterfall_fp_confident.png`
- `outputs/figures/shap_waterfall_fp_marginal.png`
- `outputs/figures/shap_waterfall_fn_marginal.png`
