# Journal d'inspection

État : inspection descriptive terminée le 13 novembre 2026. La préparation dérivée est faite. Les CSV bruts restent intacts. L'entraînement n'a pas commencé.

Les huit CSV sont dans `C:\data\cic-ids2017\raw`. Ils n'ont pas été modifiés. Le rapport chiffré est `docs/inspection_report.md`. L'analyse méthodologique, sans nettoyage, est dans `docs/methodology_report.md`. La comparaison des 21 découpages chronologiques et des 7 020 lignes contradictoires est dans `docs/protocol_comparison.md`. La vérification du parseur, de la première occurrence et du support de PR-AUC est dans `docs/protocol_verification.md`. La préparation autorisée est dans `docs/preparation_report.md`. Les CSV bruts n'ont pas été modifiés.

Mesures principales : 2 830 743 lignes, 2 273 097 Normal et 557 646 Attack, 1 358 valeurs manquantes dans `Flow Bytes/s`, 4 376 infinis positifs, 308 381 copies exactes supplémentaires. Aucune colonne de date ou d'heure. Les trois modèles prévus restent la régression logistique, la forêt aléatoire et XGBoost.
