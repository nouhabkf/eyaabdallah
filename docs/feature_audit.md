# Audit des dix variables SHAP

Le modèle XGBoost, ses paramètres, les seuils 0,50 et 0,01 et les fichiers d'évaluation du vendredi sont inchangés. Aucune variable n'est retirée. Aucun modèle n'est réentraîné. L'analyse de stabilité n'est pas lancée.

Les dix variables auditées sont celles du SHAP d'illustration déjà calculé sur 2 000 flux du vendredi. Ce classement ne désigne pas une variable suspecte. La règle préenregistrée demande deux conditions, toutes deux hors métrique de test :

1. La variable appartient au top 10 d'au moins deux méthodes parmi l'impureté de la forêt déjà entraînée, la permutation sur la validation et le SHAP sur la validation.
2. Une raison de domaine est écrite. Les raisons admissibles sont un proxy du protocole expérimental, une colonne dupliquée, un artefact de mesure, une valeur impossible, ou une distribution dégénérée sur le train.

Une variable importante reste dans le modèle tant que ces deux conditions ne sont pas réunies. Huit variables satisfont le premier décompte. Trois ont en plus une raison de domaine appuyée sur le train et le jeudi. Les cinq autres restent, parce que leur forme suit un comportement de trafic plausible.

## Méthode

L'impureté est lue dans la forêt enregistrée, sur les 61 caractéristiques, sans nouvel ajustement.

La permutation utilise la PR-AUC des scores continus du XGBoost. L'échantillon compte 5 000 flux du jeudi, graine 42, tirage proportionnel aux étiquettes d'origine, cinq répétitions. Il contient 4 973 flux normaux et 27 attaques : 19 Web Attack Brute Force et 8 XSS. Infiltration et SQL Injection n'y entrent pas. La chute moyenne de PR-AUC est 0,876 ± 0,044 pour Destination Port et 0,367 ± 0,075 pour Fwd IAT Min. Plus bas dans le top 10, les écarts-types peuvent changer l'ordre.

Le SHAP de validation reprend TreeExplainer, sur CPU, en log-odds de la classe attaque, sur 2 000 flux du jeudi. Cet échantillon contient 11 attaques : 8 Brute Force et 3 XSS. L'écart maximal de reconstruction de la marge est 8,6 × 10⁻⁶. Ce vote est fragile. Il ne remplace pas les effectifs complets utilisés pour les distributions.

Les distributions, elles, portent sur tous les flux de chaque journée. Les étiquettes Web Attack conservent le caractère U+FFFD déjà présent dans les fichiers sources ; les graphiques l'affichent par un trait d'union. Les effectifs de familles coïncident avec le manifeste. Aucune des dix colonnes n'est une copie exacte d'une autre sur le train.

## Les trois classements

| Rang | Impureté, forêt | Permutation, jeudi | SHAP, jeudi |
|---:|---|---|---|
| 1 | Packet Length Variance | Destination Port | Destination Port |
| 2 | Packet Length Std | Fwd IAT Min | Bwd Packet Length Std |
| 3 | Bwd Packet Length Std | Init_Win_bytes_forward | Bwd Packet Length Mean |
| 4 | Bwd Packet Length Max | Fwd IAT Mean | Init_Win_bytes_backward |
| 5 | Destination Port | Bwd Packet Length Mean | Init_Win_bytes_forward |
| 6 | Bwd Packet Length Mean | Init_Win_bytes_backward | Fwd IAT Min |
| 7 | Max Packet Length | Fwd IAT Max | Bwd Packets/s |
| 8 | Packet Length Mean | Flow IAT Std | Bwd Packet Length Min |
| 9 | Total Length of Bwd Packets | Flow IAT Min | Bwd Packet Length Max |
| 10 | Average Packet Size | Bwd Packet Length Min | min_seg_size_forward |

La forêt, entraînée lundi-mercredi, place d'abord les tailles de paquets. Le jeudi, où les attaques web visent un seul service, place Destination Port en tête des deux autres méthodes.

## Tableau d'audit

| Variable | Rangs forêt / permutation / SHAP jeudi | Méthodes dans le top 10 | Lecture | Expérience de retrait |
|---|---|---:|---|---|
| Destination Port | 5 / 1 / 1 | 3 | Proxy du service visé par l'outil | Oui, en premier |
| Init_Win_bytes_backward | 16 / 6 / 4 | 2 | Fenêtre du répondeur, constante selon la journée | Oui |
| Init_Win_bytes_forward | 23 / 3 / 5 | 2 | Fenêtre de la source, constante sur une partie des outils | Oui, avec la réserve ci-dessous |
| Bwd Packet Length Mean | 6 / 5 / 3 | 3 | Taille des réponses, signal plausible | Non |
| Bwd Packet Length Std | 3 / 27 / 2 | 2 | Même famille de mesures, corrélée à la moyenne | Non |
| Bwd Packet Length Max | 4 / 22 / 9 | 2 | Même famille de mesures | Non |
| Bwd Packet Length Min | 35 / 10 / 8 | 2 | Presque toujours 0 sur les attaques du train, sans être un artefact | Non |
| Fwd IAT Min | 28 / 2 / 6 | 2 | Délai minimal, cohérent avec les attaques lentes | Non |
| Bwd Packets/s | 37 / 57 / 7 | 1 | Débit dispersé, plausible | Non |
| min_seg_size_forward | 36 / 16 / 10 | 1 | Le train ne sépare pas 20 et 32 | Non |

## Destination Port

C'est le port du destinataire du flux. Dans un flux ouvert par le client, il désigne le service : 53 pour le DNS, 80 pour HTTP, 443 pour HTTPS, 21 pour FTP, 22 pour SSH. Un détecteur peut s'en servir si une attaque vise un service précis et si le trafic normal s'en distingue. Ici, chaque outil du laboratoire vise un seul port.

Sur le train, le trafic normal utilise 47 297 ports. Le plus fréquent est 53, pour 42,5 % des flux, puis 443 pour 24,0 % et 80 pour 10,3 %. Les familles d'attaque sont à l'opposé :

| Famille | Flux | Port | Part | Part de ce port dans le BENIGN |
|---|---:|---:|---:|---:|
| DoS Hulk | 172 719 | 80 | 100 % | 10,3 % |
| DoS GoldenEye | 10 286 | 80 | 100 % | 10,3 % |
| DoS slowloris | 5 384 | 80 | 100 % | 10,3 % |
| DoS Slowhttptest | 5 228 | 80 | 100 % | 10,3 % |
| FTP-Patator | 5 933 | 21 | 99,98 % | 0,16 % |
| SSH-Patator | 3 219 | 22 | 100 % | 0,33 % |
| Heartbleed | 11 | 444 | 100 % | 0 % |

95,5 % des attaques du train vont vers le port 80, contre 10,3 % du trafic normal. Le port 80 ne sépare pas à lui seul un DoS HTTP d'une navigation. Les ports 21 et 22, eux, suffisent presque à reconnaître FTP-Patator et SSH-Patator. Heartbleed confirme le même schéma, avec 11 flux seulement.

Le jeudi répète le schéma. Brute Force (1 470), XSS (652) et SQL Injection (21) vont tous vers le port 80, qui représente 10,2 % du BENIGN du jeudi. Les 36 flux d'Infiltration vont vers le port 444, présent dans 0,02 % du BENIGN. Cette dernière famille est petite. La permutation, malgré ses 27 attaques, mesure une chute de PR-AUC de 0,876 quand le port est mélangé.

Le vendredi est décrit, sans entrer dans la décision. Presque tous les flux DDoS vont vers le port 80 : quatre valeurs distinctes apparaissent, et la plus fréquente couvre plus de 99,9 % des flux. Le PortScan touche 1 000 ports, et son port le plus fréquent, 80, ne couvre que 0,4 % de ses flux : le raccourci ne s'applique pas à un balayage. Le Bot concentre 63,9 % de ses 1 953 flux sur le port 8080, qui représente 0,30 % du BENIGN du vendredi.

Raison retenue : proxy du protocole expérimental. Le port reproduit le service choisi pour chaque campagne. Aucune valeur de port n'est impossible : le minimum et le maximum restent dans 0–65535.

## Fenêtres TCP initiales

`Init_Win_bytes_backward` est le nombre d'octets de la fenêtre initiale dans le sens retour. `Init_Win_bytes_forward` est le même compte dans le sens aller. Une fenêtre TCP est positive ou nulle. La valeur −1, impossible pour une fenêtre, sert de code d'absence : elle couvre 56,0 % du BENIGN du train pour la fenêtre arrière et 44,1 % pour la fenêtre avant, et elle est absente de toutes les attaques du train dans le sens avant. Le port modal du BENIGN est 53. Le code −1 accompagne donc surtout les flux sans handshake TCP, en particulier le DNS, plutôt qu'une attaque.

La valeur présente, elle, est une constante de machine.

Sur le train, la valeur 235 représente 85,6 % des attaques et 0,17 % du BENIGN. Elle couvre 94,6 % de DoS Hulk. FTP-Patator a sa propre constante, 227, pour 67,2 % de ses flux. SSH-Patator a 247 pour 91,6 %. Le jeudi ne répète pas 235 : cette valeur ne couvre que 3,6 % de ses attaques. La constante du jeudi est 28960, pour 83,4 % des attaques et 0,37 % du BENIGN, dont 82,5 % du Brute Force et 92,8 % du XSS. La constante change avec la journée, donc avec la machine utilisée ce jour-là. Ce n'est pas une signature d'attaque stable.

`Init_Win_bytes_forward` a une preuve moins uniforme. La valeur 29200 couvre 67,0 % de FTP-Patator, 71,4 % de GoldenEye, 71,8 % de slowloris, 80,6 % de Slowhttptest et 92,5 % de SSH-Patator. Le gros du train est DoS Hulk, dont le mode est 0 pour 34,1 % des flux, pas 29200. Au total, 29200 représente 18,1 % des attaques du train et 9,5 % du BENIGN. Le jeudi est beaucoup plus tranché : 29200 couvre 92,0 % des attaques et 9,8 % du BENIGN. La réserve est donc celle-ci : la fenêtre avant identifie plusieurs outils, pas l'attaque majoritaire du train.

Les deux variables entrent dans deux top 10 officiels. Le rang SHAP du jeudi repose sur 11 attaques. Les parts citées, elles, sont calculées sur toutes les lignes. Raison retenue pour les deux : proxy du protocole expérimental. Aucune valeur inférieure à −1 n'a été trouvée.

## Longueurs de paquets retour

La moyenne, l'écart-type, le minimum et le maximum décrivent la taille des paquets qui reviennent. Une réponse volumineuse, une réponse vide ou une réponse constante sont des comportements observables. Sur 100 000 flux du train, la corrélation de Pearson vaut 0,982 entre l'écart-type et le maximum, 0,956 entre la moyenne et le maximum, 0,938 entre l'écart-type et la moyenne. Les colonnes ne sont pas identiques. Elles redisent la même taille, ce qui explique qu'elles se partagent l'importance.

DoS Hulk, 172 719 flux, a une médiane de moyenne à 1 932 octets et une médiane de maximum à 5 792. Le BENIGN du train a une médiane de moyenne à 87 octets, et 73,7 % de ses écarts-types sont nuls. Slowhttptest et slowloris sont au contraire surtout vides : 90,3 % et 66,4 % des écarts-types valent 0. Le jeudi aussi : 89,9 % des attaques web ont une moyenne nulle. Le vendredi sépare encore les familles, sans servir à la décision : la médiane de moyenne du DDoS est 1 934, celle du PortScan est 6.

Le minimum est plus pauvre. Il vaut 0 pour 99,8 % des attaques du train, et pour 48,1 % du BENIGN, dont la médiane reste 6. La colonne n'est pas dégénérée sur l'ensemble du train : le BENIGN y prend 526 valeurs. Zéro est une taille possible, pas une valeur impossible. Aucune longueur négative n'apparaît.

Ces quatre variables peuvent aider à reconnaître une réponse volumineuse ou vide. Leur importance n'en fait pas un artefact de collecte. Elles restent dans le modèle. Une expérience de retrait n'est pas proposée.

## Fwd IAT Min et Bwd Packets/s

`Fwd IAT Min` est le plus petit intervalle entre paquets dans le sens aller, en microsecondes. Un flux lent et un flux en rafale n'ont pas le même minimum. Sur le train, la médiane est 3 pour le BENIGN, 2 pour DoS Hulk, et environ 997 000 pour slowloris et Slowhttptest, soit près d'une seconde. C'est le rythme attendu de ces deux outils. Onze valeurs négatives existent sur le train, et six sur le vendredi, sur plus d'un million et demi de lignes. Elles sont impossibles pour un délai, et trop rares pour retirer la colonne. La permutation lui donne le deuxième rang, avec une chute de 0,367. Rien dans la forme des familles n'en fait un proxy de scénario.

`Bwd Packets/s` est le débit de paquets retour. Les valeurs sont très dispersées : 531 422 valeurs distinctes sur le BENIGN du train, médiane 26,7 paquets par seconde, contre une médiane de 0,070 pour DoS Hulk. Aucune constante de famille n'apparaît. La forêt la classe 37e et la permutation 57e. Son septième rang SHAP du jeudi vient de l'échantillon de 11 attaques. Elle reste une mesure de rythme plausible.

## min_seg_size_forward

C'est la plus petite taille de segment observée dans le sens aller. Elle ne prend que quelques valeurs, souvent 20, 32 ou 40, ce qui peut refléter la pile TCP de la machine source.

Sur le train, 20 représente 50,5 % du BENIGN et 42,6 % des attaques. La valeur 32 représente 46,5 % du BENIGN et 55,2 % des attaques. Les deux populations se recouvrent. Le jeudi est plus concentré : 32 couvre 98,3 % des attaques et 39,8 % du BENIGN, et 100 % du Brute Force comme du XSS. Cette concentration du jeudi ressemble à l'empreinte d'un client, mais la variable n'entre que dans le SHAP de validation. La règle demande deux méthodes. Le train ne fournit pas une valeur rare. Aucune expérience de retrait n'est proposée. Trente-cinq valeurs négatives sur le train sont des mesures impossibles, isolées.

## Ce qui est proposé, et ce qui ne l'est pas

Le cahier demande un réentraînement sans l'ensemble des variables suspectes, pas trois retraits séparés. Cette expérience groupée a été faite ensuite et validée. Elle est écrite dans `docs/project_report.md` et `docs/ablation_suspect_features.md`. Le modèle officiel n'a pas changé. Les trois retraits séparés ci-dessous n'ont pas été lancés : ils auraient servi à attribuer la baisse à une variable plutôt qu'à une autre.

1. Retirer Destination Port. C'est le proxy le plus net : un port par campagne, dès le train, répété le jeudi, et confirmé par les trois méthodes.
2. Retirer `Init_Win_bytes_backward`. La constante passe de 235 le lundi-mercredi à 28960 le jeudi, et chacune est rare dans le trafic normal du même jour.
3. Retirer `Init_Win_bytes_forward`. La même logique vaut pour 29200 sur plusieurs outils et sur le jeudi. DoS Hulk, majoritaire dans le train, ne suit pas cette constante : l'expérience est utile, elle est moins prioritaire que les deux précédentes.

Les longueurs de paquets retour, `Fwd IAT Min`, `Bwd Packets/s` et `min_seg_size_forward` restent. Les premières mesurent la réponse, le délai minimal mesure le rythme, le débit est dispersé, et la taille de segment ne sépare pas le train.

## Fichiers

- `outputs/metrics/feature_audit.json`
- `outputs/tables/feature_audit_summary.csv`
- `outputs/tables/feature_audit_by_family.csv`
- `outputs/tables/feature_audit_decision.csv`
- `outputs/figures/audit_destination_port.png`
- `outputs/figures/audit_mode_share.png`
- `outputs/figures/audit_sentinel_neg1.png`
- `outputs/figures/audit_zero_share.png`
- `outputs/figures/audit_method_ranks.png`
- `outputs/metrics/shap_values_validation_sample.npy`
