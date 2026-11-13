# Architecture

La configuration exécutable est `config/config.yaml`. `explainable_ids.config.validate_config` refuse une modification qui contredit le protocole. Les modules de `src/explainable_ids` décrivent les contrats et s'arrêtent tant que leur phase est fermée.

## Dossiers

| Chemin | Rôle |
| --- | --- |
| `config/config.yaml` | Protocole, graines, plafonds CPU |
| `src/explainable_ids/` | Contrats des étapes |
| `scripts/00_verify_setup.py` | Vérification sans entraînement |
| `scripts/01_prepare.py` | Nettoyage et découpages, sans entraînement |
| `scripts/02` à `05` | Points d'entrée encore fermés |
| `docs/` | Protocole lisible, journal, rôles |
| `tests/` | Règles de configuration et phases fermées |
| `outputs/` | Futurs métriques, figures, tables et modèles |
| `C:\data\cic-ids2017` | CSV, parquet et indices, hors OneDrive |
| `C:\venvs\explainable-ids` | Interpréteur du projet, hors OneDrive |

## Ordre

1. Nettoyage sans imputation.
2. Suppression des doublons exacts.
3. Découpage principal.
4. Recherche dans un sous-échantillon du train.
5. Réentraînement sur tout le train.
6. Choix du modèle par PR-AUC de validation.
7. Choix du seuil sur la validation.
8. Lecture unique du test.
9. Explications, sans modifier le modèle.
10. Audit sans décision fondée sur le test.
11. Stabilité sur des partitions documentées.

## Découpage

Le découpage officiel est chronologique : apprentissage du lundi au mercredi, validation le jeudi, test le vendredi. Sa validation choisira plus tard le modèle et le seuil. Son test ne choisit rien.

Le découpage groupé stratifié 70 % / 15 % / 15 %, graine 42, est le protocole complémentaire. Il ne choisit ni le modèle officiel, ni le seuil officiel.

Limites écrites avant tout résultat :

- Des flux d'une même campagne se ressemblent. Le score groupé peut être optimiste.
- Le jeudi ne contient pas les familles du vendredi. Le choix du modèle sur cette validation est un choix de transfert.
- Les répétitions de stabilité restent sur le même jeu nettoyé. Ce ne sont pas des réseaux nouveaux.

## Scores et seuil

La classe positive est l'attaque, codée 1. Le score est la probabilité `predict_proba` de cette classe.

La PR-AUC publiée est `sklearn.metrics.average_precision_score`. La métrique interne `aucpr` de XGBoost ne la remplace pas.

Pour chaque modèle, le seuil de la validation minimise `FPR + 2 * FNR`. Les taux évitent que la classe majoritaire impose le seuil. Le poids 2 du faux négatif est figé avant les résultats : une attaque manquée pèse davantage, et les fausses alertes restent dans le coût. En cas d'égalité, on garde le taux de faux négatifs le plus bas, puis le seuil le plus proche de 0,5.

Le seuil 0,5 est évalué en plus, sur la validation et sur le test. Il ne remplace pas le seuil choisi.

La validation sert aussi à désigner le modèle principal. Ses chiffres de sélection ne sont donc pas une estimation impartiale. Le test officiel est lu une fois.

## Modèles

Régression logistique avec `class_weight="balanced"`, forêt aléatoire avec `class_weight="balanced"`, XGBoost `hist` sur CPU. `scale_pos_weight` vaut `n_normal / n_attack` du train de la partition, jamais une constante écrite d'avance.

La recherche utilise trois plis stratifiés internes au sous-échantillon de train, au plus 200 000 lignes. L'arrêt anticipé est désactivé, pour ne pas consommer la validation officielle. Les hyperparamètres gagnants sont réentraînés sur tout le train.

## Explications et audit

L'impureté vient de la forêt. La permutation et le SHAP global qui servent aux décisions de variables viennent de la validation. Les cinq cas locaux viennent du test, par la règle `scores_and_labels_only_v1`, et leurs graphiques sont calculés après la sélection. Ces cas n'autorisent aucun changement.

Les critères d'une variable suspecte sont dans `docs/audit_protocol.md`.

## Stabilité

Deux familles de répétitions, toutes deux documentées dans `C:\data\cic-ids2017\splits\stability` :

- Graines de modèle 42, 7 et 123 sur la partition officielle. Seule la graine d'algorithme change.
- Graines de découpage 1001, 1002 et 1003. Chaque graine produit une partition stratifiée 70/15/15 neuve, avec ses propres indices et ses empreintes. Elle ne recopie pas le découpage officiel.

Sur chaque partition, l'imputation, la standardisation et `scale_pos_weight` sont recalculés sur son train. Le seuil est recalculé sur sa validation. Rien n'est recalculé sur son test. Les hyperparamètres et la famille du modèle principal restent ceux du run officiel. Le meilleur run de stabilité ne devient pas le résultat officiel.

## Cette livraison

La préparation des données est en place. L'entraînement des trois modèles est ouvert sur la partition chronologique. L'évaluation du test, SHAP et l'audit lèvent encore une erreur. Aucune donnée fictive n'est créée.
