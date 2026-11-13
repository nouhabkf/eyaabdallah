# Expliquer un détecteur d'intrusion

Détecteur binaire de flux réseau sur CIC-IDS2017, avec une explication des alertes. La classe positive est l'attaque. Le modèle officiel est un XGBoost appris du lundi au mercredi. Son seuil, 0,01, est choisi sur le jeudi. Le vendredi sert une seule fois à l'évaluation finale.

La question du projet est de savoir si ce détecteur apprend des comportements d'intrusion, ou des particularités de la collecte. La réponse, documentée dans le notebook, est qu'il apprend les deux. Le port de destination et les fenêtres TCP initiales portent une grande part du classement du jeudi. Les longueurs de paquets retour et le délai minimal aller restent des signaux plus plausibles.

## Résultats principaux

Sur le jeudi, la PR-AUC vaut 0,034 pour la régression logistique, 0,791 pour la forêt aléatoire et 0,932 pour XGBoost. XGBoost est retenu.

Sur le vendredi, la PR-AUC vaut 0,806. Au seuil 0,01, la précision vaut 0,997 et le rappel 0,382. Le modèle détecte une partie du DDoS, presque aucun PortScan, et aucun Bot. Au seuil 0,50, le rappel tombe à 0,028. La précision ne baisse pas lorsque le seuil passe de 0,50 à 0,01.

Retirer ensemble le port de destination et les deux fenêtres initiales fait passer la PR-AUC du jeudi de 0,932 à 0,618. Cette ablation ne remplace pas le modèle officiel. Le vendredi n'a pas été rescoré pour elle.

Les graines 42, 7 et 123 donnent le même classement SHAP, parce que `subsample` et `colsample_bytree` valent 1. Changer 30 % des flux du train officiel déplace la frontière du top 10. Le Jaccard moyen y vaut 0,818.

## Données

CIC-IDS2017 décrit cinq journées de flux, du lundi 3 juillet 2017 au vendredi 7 juillet 2017. Chaque ligne est un flux résumé par des statistiques et une étiquette de famille. Le modèle regroupe toutes les attaques dans une seule classe.

Les fichiers publics contiennent des copies exactes, une colonne d'en-tête dupliquée, des débits infinis lorsque la durée est nulle, et des profils identiques portant deux étiquettes. Engelen, Rimmer et Joosen (2021) décrivent d'autres défauts de construction et d'étiquetage, notamment les appendices TCP et des flux sans charge utile marqués comme des attaques. Le nettoyage de ce projet ne reconstruit pas les PCAP. Il retire la colonne dupliquée, exclut les profils contradictoires et ne conserve que la première occurrence chronologique de chaque contenu exact. Les infinis deviennent des valeurs manquantes. L'imputation par médiane est ajustée plus tard, sur le train seulement.

L'inspection des 2 830 743 lignes brutes compte 308 381 copies exactes supplémentaires et 1 358 jetons `NaN`. Le fichier dérivé, après nettoyage, conserve 2 520 966 lignes. Il retire 302 757 copies, une fois les 7 020 lignes contradictoires exclues, et il conserve 351 jetons manquants.

| Ensemble | Jours | Lignes |
|---|---|---:|
| Apprentissage | lundi–mercredi | 1 527 133 |
| Validation | jeudi | 398 110 |
| Test | vendredi | 595 723 |

## Modèles et explications

Trois modèles sont entraînés, sans SMOTE.

- Régression logistique, `class_weight="balanced"`, après imputation et standardisation.
- Forêt aléatoire, `class_weight="balanced"`.
- XGBoost sur processeur, `tree_method=hist`. `scale_pos_weight` vaut le rapport du trafic normal aux attaques du train.

Les hyperparamètres sont cherchés par validation croisée à trois plis dans un sous-échantillon du train. Le jeudi et le vendredi n'y entrent pas. Le seuil minimise `FPR + 2 × FNR` sur le jeudi. Le seuil 0,5 est aussi rapporté.

Trois lectures d'importance sont comparées. L'impureté vient de la forêt. La permutation mesure la baisse de PR-AUC sur tout le vendredi, cinq répétitions, graine 42. Le SHAP global est la moyenne des valeurs absolues sur 2 000 flux du vendredi, graine 42, en log-odds de l'attaque. Cinq flux sont commentés : deux vrais positifs, deux faux positifs et un faux négatif.

## Limites

Le jeudi ne contient pas les familles du vendredi. Le choix du modèle et du seuil n'est donc pas une estimation impartiale du test. Le seuil 0,01 est au bord de la grille. Des variables corrélées se partagent l'importance. L'ablation retire trois variables à la fois, donc la baisse de PR-AUC ne se répartit pas entre elles. Les défauts de CICFlowMeter décrits par Engelen et al. restent dans le fichier dérivé.

## Installation

Python 3.12. Les versions utilisées pour l'entraînement sont fixées dans `requirements.txt`.

```powershell
python -m venv C:\venvs\explainable-ids
C:\venvs\explainable-ids\Scripts\python.exe -m pip install -r requirements.txt
```

Bibliothèques principales : numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.54.1, matplotlib 3.11.2, pyarrow 26.0.1, joblib 1.6.0, jupyter 1.1.1.

Graines. `PYTHONHASHSEED` vaut 42. La recherche, les modèles officiels, l'échantillon SHAP et la permutation utilisent la graine 42. L'étude de stabilité ajoute les graines de modèle 7 et 123, puis les graines 1001, 1002 et 1003 pour les sous-échantillons du train.

## Données CIC-IDS2017

La source publique est le jeu *Intrusion Detection Evaluation Dataset* de l'Université du Nouveau-Brunswick :

https://www.unb.ca/cic/datasets/ids-2017.html

Il faut les huit CSV de flux (*MachineLearningCSV*), pas les PCAP. Ils ne sont pas versionnés. Les placer, sans les modifier, dans :

```text
C:\data\cic-ids2017\raw
```

Le fichier dérivé produit par `scripts/01_prepare.py` est écrit dans `C:\data\cic-ids2017\interim\flows.parquet`. Les indices de découpage vont dans `C:\data\cic-ids2017\splits`.

Le notebook de rapport n'a pas besoin de relire ces CSV. Il charge les métriques, tableaux et figures déjà enregistrés dans `outputs/`.

## Notebook

Le rapport complet est `notebooks/expliquer_un_detecteur_intrusion.ipynb`. Depuis la racine du projet :

```powershell
$env:PYTHONHASHSEED = "42"
$env:PYTHONIOENCODING = "utf-8"
C:\venvs\explainable-ids\Scripts\python.exe -m jupyter notebook notebooks/expliquer_un_detecteur_intrusion.ipynb
```

L'exécution affiche les résultats enregistrés. Elle ne réentraîne pas les modèles et ne recalcule pas SHAP. Pour relancer la chaîne depuis les CSV, les scripts `scripts/00_verify_setup.py` à `scripts/08_test_permutation.py` suivent l'ordre du protocole. Les phases déjà exécutées ne doivent pas être relancées si l'on veut conserver les fichiers officiels.

## Organisation

```text
config/          protocole et graines
src/explainable_ids/  code du pipeline
scripts/         points d'entrée, de la vérification à la permutation
tests/           contrôles du protocole
docs/            rapports de chaque étape
notebooks/       rapport final exécutable
outputs/metrics  JSON et scores enregistrés
outputs/tables   tableaux CSV
outputs/figures  courbes, matrices et graphiques SHAP
outputs/models   modèles officiels et expériences séparées
requirements.txt versions des bibliothèques
```

Les modèles d'expérience, ablation et stabilité, sont des fichiers distincts. Ils ne remplacent pas `outputs/models/xgboost.joblib`.
