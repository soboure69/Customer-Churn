# Customer-Churn
Projet de machine learning visant à prédire le désabonnement des clients d'une entreprise de télécommunications, également appelé "customer churn".
![alt text](image_customer_churn.png)
# Projet de Machine Learning : Prédiction du Désabonnement Client (Customer Churn)

Ce projet vise à construire un modèle de Machine Learning pour **prédire le désabonnement des clients (Customer Churn)** au sein d'une entreprise de télécommunication. La prédiction du taux de désabonnement est cruciale pour toute entreprise, car **le coût de rétention d'un client existant est bien inférieur à celui de l'acquisition d'un nouveau client**.

L'objectif est de permettre à l'entreprise de cibler spécifiquement les clients à risque élevé de désabonnement avec des campagnes marketing et des offres intéressantes pour les retenir et les fidéliser.

## 1. Introduction et Problématique Business

Le **désabonnement d'un client** survient lorsque ce dernier décide de cesser d'utiliser les services, le contenu ou les produits d'une entreprise . Cela peut inclure l'annulation d'un service sous contrat, la rupture d'une relation non contractuelle, le choix d'un concurrent, la désinscription d'une newsletter ou la clôture d'un compte bancaire.

Dans un monde des affaires de plus en plus concurrentiel, le **coût d'acquisition de nouveaux clients est très élevé** . Par conséquent, la **fidélisation des clients existants est plus importante** pour les entreprises. Comprendre le comportement des clients est essentiel pour les fidéliser. La création d'un modèle d'apprentissage automatique ou de réseaux de neurones artificiels peut prédire les clients susceptibles de se désabonner, permettant ainsi des actions de rétention ciblées.

## 2. Étapes du Projet

Le projet suit plusieurs étapes structurées:

- **Introduction** (comprendre la problématique business)
- **Importation des outils nécessaires** (Python et librairies)
- **Importation des données**
- **Analyse exploratoire des données** (mieux comprendre les données et identifier les traitements nécessaires)
- **Prétraitement des données** (nettoyage et préparation)
- **Modélisation** (entraînement de plusieurs algorithmes, comparaison et sélection du meilleur modèle)
- **Conclusion**

## 3. Importation des Outils et Données

Les outils principaux utilisés sont le langage **Python** et des librairies essentielles pour la science des données et le Machine Learning:

- **Pandas**: Pour l'analyse et la manipulation de données (notamment les DataFrames).
- **NumPy**: Pour les calculs numériques rapides et scientifiques.
- **Seaborn et Matplotlib**: Pour la création de visualisations graphiques.
- **IPywidgets**: Pour créer des graphiques interactifs dans les notebooks.
- **Scikit-learn**: Pour les algorithmes de Machine Learning, la normalisation des données, la division des jeux de données, le ré-échantillonnage, et les métriques d'évaluation.

Les données proviennent d'une entreprise de télécommunication et peuvent être téléchargées depuis **[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)**.

L'ensemble de données initial contient **7043 observations (clients) et 21 colonnes (variables)**, avec la variable cible `Churn` indiquant si un client s'est désabonné (`Yes`) ou non (`No`). C'est un **problème de classification binaire**.

## 4. Analyse Exploratoire des Données (AED)

L'AED a pour but de mieux comprendre les données et d'identifier les problèmes à résoudre.

- **Analyse Univariée**:
    - **Variables Catégorielles**: Création de graphiques à barres interactifs pour visualiser la distribution de chaque variable catégorielle (celles avec moins de 5 modalités distinctes).
        - Observation d'un **déséquilibre de classe** pour la variable cible `Churn`: environ **73% de clients non désabonnés (No) contre 27% de désabonnés (Yes)**. Ce déséquilibre peut impacter négativement la performance des modèles.
        - Distribution à peu près égale pour la variable `Gender` (environ 50/50).
        - Moins de clients âgés (`SeniorCitizen`) que de jeunes.
    - **Variables Numériques (Quantitatives Continues)**: Création d'histogrammes interactifs.
        - Identification d'un problème avec la variable `TotalCharges` (montant total facturé) qui est reconnue comme une chaîne de caractères par Pandas en raison d'espaces (valeurs manquantes cachées).
        - La variable `TotalCharges` présente une **forte asymétrie positive (coefficient d'asymétrie de 0.96)**, ce qui peut causer des problèmes pour certains algorithmes.
        - Les boîtes à moustaches n'ont pas révélé de valeurs aberrantes significatives.

- **Analyse Bivariée**:
    - **`TotalCharges` vs `Churn`**: Le montant total facturé est en moyenne **plus faible pour les clients désabonnés**.
    - **`MonthlyCharges` vs `Churn`**: Les clients désabonnés sont en moyenne **plus facturés mensuellement** que les clients non désabonnés. Cela suggère que des facturations excessives peuvent être un facteur de désabonnement.
    - **`Tenure` vs `Churn`**: Les clients non désabonnés restent fidèles en moyenne beaucoup plus longtemps (environ 40 mois) que les clients désabonnés (moins de 10 mois).
    - **`MonthlyCharges` par statut `Churn` et `SeniorCitizen`**: Les personnes âgées sont **plus facturées mensuellement** que les jeunes dans les deux cas (désabonnés ou non).
    - **`MonthlyCharges` par statut `Churn` et `Dependents`**: Les clients sans personne à charge qui ne se sont pas désabonnés sont plus facturés.

- **Résumé Statistique**:
    - Les variables continues (`Tenure`, `MonthlyCharges`, `TotalCharges`) ont des **échelles très différentes**. De nombreux modèles de Machine Learning fonctionnent mieux avec des variables standardisées ou normalisées.

## 5. Prétraitement des Données

Cette phase est cruciale pour nettoyer et préparer les données pour la modélisation, représentant **70% à 80% du temps d'un projet de data scientist**.

- **Gestion des Valeurs Manquantes**:
    - Détection de **11 valeurs manquantes cachées** dans `TotalCharges` (représentées par des espaces).
    - Remplacement de ces espaces par des valeurs `NaN` et conversion de la colonne en type numérique (float).
    - **Suppression des 11 lignes** contenant ces valeurs manquantes, car elles ne représentent qu'une infime partie des données (< 0.05%).

- **Encodage des Variables Catégorielles**:
    - Les algorithmes de Machine Learning de Scikit-learn nécessitent des valeurs numériques.
    - **Variables Binaires (2 modalités)**: Encodage manuel ou semi-automatique (ex: `Gender` Mâle/Femelle en 1/0, `Yes`/`No` en 1/0).
    - **Variables Catégorielles Multi-modalités**: Utilisation de la fonction `get_dummies` de Pandas pour le **One-Hot Encoding**, en supprimant une des modalités (`drop_first=True`) pour éviter la multi-colinéarité.
    - Concaténation de toutes les variables encodées pour former une nouvelle DataFrame prête à la modélisation.

- **Transformation des Variables Asymétriques**:
    - Application d'une **fonction racine carrée** à la variable `TotalCharges` pour réduire son asymétrie. Le coefficient d'asymétrie passe de 0.96 à environ 0.31, améliorant significativement la distribution.

- **Division des Données**:
    - Division de l'ensemble de données en trois parties pour un test de généralisation robuste du modèle:
        - **Données d'entraînement (60%)**: Pour entraîner les algorithmes.
        - **Données de validation (20%)**: Pour sélectionner le meilleur modèle parmi les différents algorithmes entraînés.
        - **Données de test (20%)**: Pour évaluer la capacité de généralisation du meilleur modèle sur des données totalement inédites.
    - Utilisation du paramètre `stratify=y` dans `train_test_split` pour assurer que la **proportion des classes (désabonnés/non-désabonnés) est la même dans tous les sous-ensembles**. Ceci est crucial en raison du problème de déséquilibre de classe.

- **Gestion du Déséquilibre de Classe**:
  - Le déséquilibre (73% non-churn, 27% churn) peut biaiser les modèles. Deux techniques de **ré-échantillonnage** sont abordées:
        - **Sur-échantillonnage (Oversampling)**: Créer plus d'observations dans la classe minoritaire (`Yes`) pour atteindre l'équilibre (ex: `RandomOverSampler`). Dans ce projet, on a choisi de travailler avec des données sur-échantillonnées par défaut.
        - **Sous-échantillonnage (Undersampling)**: Diminuer artificiellement la classe majoritaire (`No`) pour atteindre l'équilibre (ex: `RandomUnderSampler`).
  -   Le choix entre ces méthodes impacte les performances et doit être testé de manière itérative.

- **Normalisation des Données**:
    - Les variables continues ayant des échelles différentes (ex: `Tenure` vs `TotalCharges`) peuvent biaiser les modèles.
    - **Normalisation (Min-Max Scaling)**: Mise à l'échelle des valeurs dans l'intervalle. Dans ce projet, on a choisi cette méthode.
    - **Standardisation (Z-score normalization)**: Remplacer les valeurs par leur Z-score (valeur - moyenne / écart-type).
    - Le choix entre normalisation et standardisation est empirique et doit être déterminé en comparant l'impact sur les performances du modèle.

## 6. Modélisation et Sélection de Variables

Cette phase consiste à entraîner divers algorithmes, évaluer leurs performances, et sélectionner le meilleur modèle.

- **Choix de la Métrique d'Évaluation**:
    - Indispensable avant d'entraîner les algorithmes.
    - **Matrice de Confusion**: Permet de visualiser les Vrais Positifs (TP), Vrais Négatifs (TN), Faux Positifs (FP) et Faux Négatifs (FN).
        - `Positif` = Client s'est désabonné (classe 1).
        - `Négatif` = Client ne s'est pas désabonné (classe 0).
    - **Précision Globale (Accuracy)**: Proportion de prévisions correctes. **Peu fiable en cas de déséquilibre de classe** car un modèle peut obtenir une bonne précision en prédisant toujours la classe majoritaire.
    - **Précision (Precision)**: Proportion de vrais positifs parmi toutes les prédictions positives (TP / (TP + FP)).
    - **Rappel (Recall)**: Capacité du modèle à identifier tous les vrais positifs (TP / (TP + FN)).
    - **F1-score**: **Moyenne harmonique de la précision et du rappel**. C'est la métrique choisie pour ce projet car elle combine les deux, offrant une évaluation plus équilibrée en cas de déséquilibre de classe. Un F1-score de 1 est parfait, 0 est très mauvais. Le choix de la métrique dépend de la problématique (ex: détection de cancer vs fraude).

- **Sélection des Variables Prédictrices**:
    - Il est important de ne garder que les variables pertinentes pour réduire le bruit et éviter le sur-ajustement.
    - **Importance des Variables (Random Forest)**: Utilisation d'un simple modèle de Forêt Aléatoire pour déterminer l'importance de chaque variable prédictive. Les variables les plus importantes étaient `TotalCharges`, `Tenure`, et `MonthlyCharges`. Des variables comme `Gender` et `SeniorCitizen` ont montré une importance très faible. Un seuil d'importance (ex: > 0.04) peut être appliqué pour sélectionner les variables finales.
    - **Recursive Feature Elimination (RFE)**: Une méthode de réduction de dimensionnalité qui classe les prédicteurs par ordre d'importance en éliminant récursivement les moins importants. Par défaut, RFE sélectionne la moitié des variables si non spécifié. La RFE permet de **réduire la complexité du modèle tout en maintenant, voire améliorant, sa performance**, et facilite l'interprétabilité.

- **Algorithmes de Machine Learning Entraînés**:
    Les algorithmes ont été entraînés avec les données d'entraînement (sur-échantillonnées), et leurs hyperparamètres optimaux ont été recherchés via **GridSearchCV** (utilisant F1-score comme métrique) et la cross-validation (5 plis).

    1. **Régression Logistique (Logistic Regression)** :
        - Un des hyperparamètres clés est `C` (inverse de la force de régularisation).
        - Meilleur score F1 sur les données d'entraînement: 0.79.
        - Performance sur données de validation (F1-score classe positive): 0.61.
        - Application de RFE a réduit le nombre de prédicteurs de 28 à 14, avec des performances similaires, rendant le modèle plus interprétable.

    2. **Forêt Aléatoire (Random Forest Classifier)**:
        - Hyperparamètres importants: `n_estimators` (nombre d'arbres) et `max_depth` (profondeur maximale des arbres).
        - Meilleur score F1 sur les données d'entraînement: 0.906.
        - Performance sur données de validation (F1-score classe positive): 0.54. Semble moins efficace que la Régression Logistique dans ce cas.
        - L'application de RFE a également réduit les variables à 14, mais n'a pas amélioré significativement la performance.

    3. **Classifieur MLP (Multilayer Perceptron Classifier - Réseau de Neurones Artificiel simple)**:
        - Hyperparamètres clés: `hidden_layer_sizes` (nombre de neurones par couche) et `activation` (fonction d'activation).
        - Meilleur score F1 sur les données d'entraînement: 0.856.
        - Performance sur données de validation (F1-score classe positive): 0.56. Toujours moins performant que la Régression Logistique sur la classe positive.

    4. **Support Vector Machine (SVM - SVC)**:
        - Hyperparamètres importants: `C` (régularisation) et `kernel` (transforme les données non linéairement séparables).
        - Meilleur score F1 sur les données d'entraînement: 0.847.
        - Performance sur données de validation (F1-score classe positive): 0.51. Aussi moins performant que la Régression Logistique.

- **Sélection et Évaluation du Meilleur Modèle**:
    - Après avoir comparé les performances sur les **données de validation**, le **modèle de Régression Logistique** a montré la meilleure performance (F1-score de 0.61 sur la classe positive).
    - Ce meilleur modèle a ensuite été évalué sur les **données de test** (jamais vues) pour confirmer sa capacité de généralisation.
    - La performance sur les données de test (F1-score de 0.64) était relativement proche de celle sur les données de validation, avec même une légère amélioration.

## 7. Conclusion et Perspectives

Ce projet a démontré la capacité à construire un modèle de Machine Learning pour la prédiction du désabonnement client.

**Compétences acquises durant le projet**:

- Comprendre une problématique business et la transformer en projet de Data Science.
- Effectuer une analyse exploratoire des données (univariée, multivariée, statistiques).
- Gérer les valeurs manquantes.
- Transformer les variables asymétriques.
- Encoder les variables catégorielles.
- Gérer le problème de déséquilibre de classe (sur-échantillonnage, sous-échantillonnage).
- Diviser les données en ensembles d'entraînement, validation et test.
- Normaliser les données.
- Construire plusieurs algorithmes de Machine Learning (Régression Logistique, Forêt Aléatoire, MLP, SVM).
- Sélectionner les variables importantes.
- Évaluer la performance des modèles en choisissant la métrique appropriée (F1-score).
- Sélectionner le meilleur modèle.

Le processus de construction d'un modèle de Machine Learning n'est **pas linéaire mais cyclique et itératif**. Pour améliorer les performances du modèle, plusieurs explorations supplémentaires peuvent être effectuées:

- **Changer la méthode de mise à l'échelle**: Tester la standardisation (StandardScaler) au lieu de la normalisation.
- **Utiliser différentes données d'entraînement**: Tester les données originelles (avec déséquilibre) ou celles obtenues par sous-échantillonnage.
- **Modifier le seuil de sélection des variables**: Essayer d'autres seuils pour la suppression des variables moins importantes.
- **Régler davantage d'hyperparamètres**: Explorer d'autres hyperparamètres et plages de valeurs pour chaque algorithme, comme le `learning_rate` pour le MLP.
- **Tester d'autres algorithmes**: Explorer d'autres types de modèles de classification.
- **Optimiser les hyperparamètres avec RFE**: Affiner la sélection de variables via RFE.

Une fois satisfait du modèle, il peut être présenté à l'entreprise pour être mis en production. Ce modèle servira alors à l'équipe marketing pour **cibler les clients à risque de désabonnement avec des offres spécifiques**, permettant à l'entreprise de **dépenser moins en acquisition de nouveaux clients et de fidéliser sa clientèle existante**. C'est une application pratique de la science des données et du Machine Learning pour résoudre une problématique business concrète.

---