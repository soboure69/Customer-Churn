# Customer Churn

Prédiction du désabonnement de clients d'un opérateur télécom, à partir du jeu public de 7 043 clients et 21 variables. La cible `Churn` est binaire. Après retrait de 11 montants `TotalCharges` illisibles, il reste 7 032 clients, dont 26,5 % sont partis.

![Illustration du désabonnement client](image_customer_churn.png)

## Résultat

Le modèle retenu est une **régression logistique** (`C = 10`), entraînée sur un train sur-échantillonné et choisie sur un jeu de validation stratifié. Graine : `111`.

| Jeu | F1 churn | Précision | Rappel | Exactitude |
|---|---:|---:|---:|---:|
| Validation (n = 1 406) | 0,622 | 0,530 | 0,751 | 0,757 |
| Test (n = 1 407) | 0,628 | 0,513 | 0,810 | 0,745 |

Les trois autres modèles, au F1 de validation : forêt aléatoire 0,551, MLP 0,519, SVM 0,508.

Sur le test, deux références cadrent ce score :

- un tirage aléatoire calé sur la proportion de churn du train donne un F1 d'environ **0,26** ;
- la règle « contrat mensuel = churn » donne un F1 de **0,564**.

La logistique apporte environ **6 points de F1** par rapport à cette règle. L'exactitude de 0,75 est à deux points d'une prédiction constante « le client reste » (0,735). Avec un rappel de 0,81 et une précision de 0,51, environ un client signalé sur deux part réellement.

## Pipeline

1. Les 11 `TotalCharges` vides (0,16 % des lignes) sont supprimés, puis la colonne est passée en numérique et transformée par racine carrée.
2. `gender` vaut 1 pour `Female`. `SeniorCitizen` reste 0/1. Les autres variables oui/non sont encodées en 0/1. Les variables à plus de deux modalités passent en indicatrices (`drop_first`).
3. Découpage stratifié 60 % / 20 % / 20 %. Le sur-échantillonnage de la classe churn ne touche que le train (`sklearn.utils.resample`).
4. `MinMaxScaler` ajusté sur le train. Un seuil d'importance de 0,004, calculé par une forêt à graine fixe, conserve les 30 prédicteurs.
5. Grille `GridSearchCV` (F1, 5 plis) pour la logistique, la forêt, le MLP et le SVM. La régression logistique est réévaluée une fois sur le test, puis enregistrée.

`gender` et `SeniorCitizen` entrent dans le modèle. Sur les variables normalisées, leurs coefficients valent environ +0,13 et +0,27. Une exécution antérieure les avait mises à zéro pour tous les clients à cause d'un encodage erroné.

La validation croisée affichée pendant la grille (F1 0,774 pour la logistique) est calculée après sur-échantillonnage. Elle ne mesure pas la performance sur des clients nouveaux. Le chiffre à retenir est le F1 de validation, 0,622.

## Prédire

Le fichier `model_pipeline.joblib` contient le scaler, l'ordre des colonnes et la régression logistique. Il permet de scorer un CSV sans relancer l'entraînement.

```bash
python predict.py --input data.csv --output preds.csv --pipeline model_pipeline.joblib
```

La sortie contient `customerID` lorsqu'il est présent, `prediction` (0/1) et `proba_churn`. Les lignes dont `TotalCharges` est illisible sont écartées et signalées, sans décaler les identifiants. `predict_from_dataframe` est importée depuis `predict.py` par la dernière cellule du notebook.

Dépendances figées dans `requirements.txt` (Python 3.13, scikit-learn 1.7.1, pandas 2.3.1).

## Limites

- Le jeu est une photo : `tenure` et `TotalCharges` des clients partis s'arrêtent à leur départ. Le modèle décrit ce snapshot. Il ne date pas une prédiction avant une fenêtre de churn.
- `TotalCharges` est presque égal à `tenure × MonthlyCharges` (corrélation 0,9996). Plusieurs colonnes « No internet service » sont identiques. La RFE réduit le modèle de 30 à 15 variables (F1 de validation 0,623), sans rendre chaque coefficient lisible isolément.
- Le seuil de décision reste 0,5. Il n'est pas choisi selon le coût d'une offre et le coût d'un départ manqué.
- Relancer le notebook de bout en bout prend environ **75 minutes**, dont environ **66** pour la grille SVM (`probability=True`).

Une validation croisée avec `class_weight='balanced'`, sans sur-échantillonnage préalable, et le réglage du seuil sont la prochaine étape. Ils ne sont pas dans ce gel, pour ne pas mélanger le correctif et un nouveau choix d'hyperparamètres.

## Sources

- Données : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (jeu IBM / Kaggle).
- Description des colonnes : `Description_Des_Variables.md`.
- `machine_learning_map.png` est une traduction de l'aide-mémoire scikit-learn, sous licence BSD : <https://scikit-learn.org/stable/machine_learning_map.html>.
