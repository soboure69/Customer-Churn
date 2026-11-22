# Customer Churn

Prédiction du désabonnement de clients d'un opérateur télécom, à partir du jeu public de 7 043 clients et 21 variables. La cible `Churn` est binaire. Après retrait de 11 montants `TotalCharges` illisibles, il reste 7 032 clients, dont 26,6 % sont partis.

![Illustration du désabonnement client](image_customer_churn.png)

## Résultat

Le modèle retenu est une **régression logistique à classes pondérées** (`class_weight='balanced'`, `C = 1`), avec un seuil de décision de **0,5**. Elle est choisie par validation croisée sans fuite sur le train, puis confirmée sur un jeu de validation stratifié. Graine : `111`.

| Jeu | F1 churn | Précision | Rappel | Exactitude |
|---|---:|---:|---:|---:|
| Validation (n = 1 406) | 0,619 | 0,524 | 0,757 | 0,752 |
| Test (n = 1 407) | 0,628 | 0,513 | 0,810 | 0,745 |

L'intervalle de confiance à 95 % du F1 de validation (bootstrap) va de 0,581 à 0,654.

Sur le test, deux références cadrent ce score :

- un tirage aléatoire calé sur la proportion de churn du train donne un F1 d'environ **0,27** ;
- la règle « contrat mensuel = churn » donne un F1 de **0,564**.

La logistique apporte environ **6 points de F1** par rapport à cette règle. L'exactitude de 0,75 est à deux points d'une prédiction constante « le client reste » (0,734). Avec un rappel de 0,81 et une précision de 0,51, environ un client signalé sur deux part réellement.

## Comparaison des modèles

Chaque modèle est un pipeline : normalisation, traitement du déséquilibre **dans chaque pli**, puis modèle. Grille `GridSearchCV`, F1 de la classe churn, validation croisée stratifiée à 5 plis sur le train d'origine.

| Modèle | Réglage retenu | F1 CV (± écart-type) | F1 validation |
|---|---|---:|---:|
| Forêt aléatoire pondérée | profondeur 10, feuilles ≥ 5, 300 arbres | 0,642 ± 0,020 | 0,622 |
| MLP, sur-échantillonnage dans les plis | couche de 200, `alpha = 0,01` | 0,634 ± 0,013 | 0,584 |
| **Régression logistique pondérée** | `C = 1` | 0,631 ± 0,012 | 0,619 |
| Régression logistique, sur-échantillonnage dans les plis | `C = 10` | 0,630 ± 0,012 | 0,622 |
| SVM pondéré | `C = 1`, noyau rbf | 0,622 ± 0,015 | 0,621 |

En validation, quatre modèles sont entre 0,619 et 0,622. Le MLP descend à 0,584, alors que sa validation croisée est au niveau des autres, et son réglage change d'une exécution à l'autre (`alpha = 0,01` ou `1`). Ces écarts restent de l'ordre de l'incertitude du F1 de validation (environ ± 0,035) : aucun modèle n'est meilleur de façon démontrable. La régression logistique est retenue parce qu'elle est la plus simple et que ses coefficients se lisent.

Les variables qui augmentent le plus le risque sont la fibre optique, le paiement par chèque électronique, les services de streaming et la facture sans papier. Celles qui le diminuent le plus sont `TotalCharges`, les contrats d'un ou deux ans et l'ancienneté. Ces coefficients décrivent des associations dans ce jeu, pas des causes.

## Seuil de décision

Le seuil est étudié sur des probabilités **hors pli** du train : chaque client est noté par un modèle qui ne l'a pas vu. Le F1 est presque plat entre 0,50 et 0,65. Le choix du seuil est donc une décision métier.

| Seuil | F1 | Précision | Rappel | Clients signalés |
|---:|---:|---:|---:|---:|
| 0,35 | 0,599 | 0,452 | 0,888 | 52 % |
| 0,50 | 0,631 | 0,525 | 0,792 | 40 % |
| 0,55 | 0,633 | 0,546 | 0,754 | 37 % |
| 0,65 | 0,628 | 0,609 | 0,648 | 28 % |

**Selon la capacité de contact** (on contacte les clients les plus risqués) :

| Clients contactés | Seuil équivalent | Précision | Rappel |
|---:|---:|---:|---:|
| 10 % | 0,85 | 0,763 | 0,287 |
| 20 % | 0,75 | 0,686 | 0,517 |
| 30 % | 0,62 | 0,599 | 0,676 |
| 40 % | 0,50 | 0,525 | 0,791 |

**Selon le rapport des coûts.** Avec `c_offre` le coût d'une offre et `c_départ` la valeur perdue quand un client part sans avoir été contacté, le seuil de coût minimal dépend seulement du rapport `c_départ / c_offre`. Aucun coût réel n'est connu : les rapports ci-dessous sont des hypothèses.

| `c_départ / c_offre` | Seuil de coût minimal | Clients contactés | Précision | Rappel |
|---:|---:|---:|---:|---:|
| 2 | 0,77 | 19 % | 0,701 | 0,492 |
| 3 | 0,63 | 30 % | 0,601 | 0,672 |
| 5 | 0,47 | 43 % | 0,511 | 0,819 |
| 10 | 0,22 | 63 % | 0,399 | 0,948 |

Le seuil de 0,5 correspond à un rapport d'environ 5, ou à une capacité de contact de 40 % des clients. Pour l'ajuster, il faut estimer `c_départ ≈ taux de succès de l'offre × valeur du client sur la période`, et le coût d'une offre.

## Pipeline

1. Les 11 `TotalCharges` vides (0,16 % des lignes) sont supprimés, puis la colonne est passée en numérique et transformée par racine carrée.
2. `gender` vaut 1 pour `Female`. `SeniorCitizen` reste 0/1. Les autres variables oui/non sont encodées en 0/1. Les variables à plus de deux modalités passent en indicatrices (`drop_first`).
3. Découpage stratifié 60 % / 20 % / 20 %.
4. Étape 2 du notebook : `Pipeline` (`MinMaxScaler` puis modèle) réglé par validation croisée sur le train d'origine. Le déséquilibre est traité dans chaque pli, par pondération des classes ou par `RandomOverSampler` d'`imbalanced-learn`.
5. Le seuil est fixé sur les probabilités hors pli du train. Le modèle final est évalué une fois sur le test, puis enregistré.

Les sections précédentes du notebook gardent la première approche : sur-échantillonnage de tout le train avant la validation croisée. Son F1 de grille (0,77 pour la logistique) était gonflé, parce que des copies d'un même client se trouvaient dans les plis d'entraînement et d'évaluation. Sans cette fuite, la validation croisée donne 0,62 à 0,64, au niveau de la validation.

`gender` et `SeniorCitizen` entrent dans le modèle. Une exécution antérieure les avait mises à zéro pour tous les clients à cause d'un encodage erroné.

## Prédire

Le fichier `model_pipeline.joblib` contient le scaler, l'ordre des colonnes, la régression logistique et le seuil. Il permet de scorer un CSV sans relancer l'entraînement.

```bash
python predict.py --input data.csv --output preds.csv --pipeline model_pipeline.joblib
```

La sortie contient `customerID` lorsqu'il est présent, `prediction` (0/1, soit `proba_churn ≥ seuil`) et `proba_churn`. Les lignes dont `TotalCharges` est illisible sont écartées et signalées, sans décaler les identifiants. La dernière cellule du notebook vérifie que `predict.py` retrouve le F1 du test.

Dépendances figées dans `requirements.txt` (Python 3.13, scikit-learn 1.7.1, imbalanced-learn 0.14.0, pandas 2.3.1).

## Limites

- Le jeu est une photo : `tenure` et `TotalCharges` des clients partis s'arrêtent à leur départ. Le modèle décrit ce snapshot. Il ne date pas une prédiction avant une fenêtre de churn.
- `TotalCharges` est presque égal à `tenure × MonthlyCharges` (corrélation 0,9996). Plusieurs colonnes « No internet service » sont identiques. Les coefficients ne décrivent donc pas chacun un effet séparé.
- Le seuil de 0,5 ne repose sur aucun coût réel. Les tableaux ci-dessus servent à le déplacer quand ces coûts seront connus.
- Le jeu de test a été consulté pendant la première approche et pendant la préparation de l'étape 2. Son score n'a servi à aucun choix, mais ce n'est plus un test entièrement neuf.
- Relancer le notebook de bout en bout prend environ 27 minutes, dont environ 13 pour la grille SVM de la première approche et 5 pour la comparaison de l'étape 2 (mesuré sur une machine à 2 cœurs).

## Sources

- Données : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (jeu IBM / Kaggle).
- Description des colonnes : `Description_Des_Variables.md`.
- `machine_learning_map.png` est une traduction de l'aide-mémoire scikit-learn, sous licence BSD : <https://scikit-learn.org/stable/machine_learning_map.html>.
