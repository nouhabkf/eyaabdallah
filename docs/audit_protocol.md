# Protocole d'audit

Ces critères sont fixés avant l'entraînement et avant toute lecture du test. Les changer après avoir vu le test invalide l'audit.

## Dix variables

Le rapport présente les dix premières variables selon trois méthodes :

1. Importance d'impureté de la forêt aléatoire.
2. Importance par permutation sur la validation, avec la PR-AUC comme score.
3. Moyenne des valeurs absolues SHAP sur un échantillon de validation.

Le test n'entre dans aucun de ces trois classements.

## Variable suspecte

Une variable est candidate seulement si les deux conditions suivantes sont vraies.

1. Elle appartient au top 10 d'au moins deux des trois méthodes.
2. Une raison de domaine est écrite avant l'ablation. Les raisons admissibles sont : proxy du protocole expérimental, colonne dupliquée, artefact de mesure, valeurs impossibles, distribution dégénérée observée sur le train.

Le texte de décision ne cite aucune métrique de test.

Si aucune variable ne remplit les deux conditions, le résultat est : aucune variable suspecte. Aucune ablation n'est forcée pour obtenir le nombre demandé.

## Ablation

L'ablation conserve les hyperparamètres du modèle officiel. Elle ne devient pas le score officiel, même si son test est meilleur. La comparaison de validation est lue en premier. La comparaison de test est un compte rendu unique et ne revient pas sur la décision.

Le modèle principal continue d'inclure la variable jusqu'à cet audit. Le retrait anticipé, pendant le nettoyage, est réservé aux colonnes identiques ou de variance nulle sur le train, et aux identifiants ou à l'horodatage.

## Stabilité des explications

Pour chaque répétition documentée, le top 10 est comparé par indice de Jaccard. Les valeurs sont rapportées. Aucun seuil de réussite n'est fixé dans la configuration, afin de ne pas en choisir un après avoir vu les chiffres.

Les scores rapportés avec la dispersion sont la PR-AUC, le rappel au seuil de la partition et le F1 à ce même seuil.
