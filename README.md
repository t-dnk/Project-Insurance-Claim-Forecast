# Project-Insurance-Claim-Forecast
## Prédiction des coûts de sinistres en assurance santé : comparaison GLM et Neural Network
 En assurance santé, l'estimation du coût des prestations est un enjeu important pour les compagnies d'assurance, car cela permet de mieux anticiper les dépenses liées aux sinistres et peut contribuer aux travaux de tarifications.
 L'objectif de ce projet est de prédire les montants individuels de sinistre `claim` à partir des caractéristiques des assurés en comparant différentes méthodes de modélisation.
 Trois approches ont été étudiées lors de ce projet : une régression linéaire multiple, un GLM Gamma avec un lien logarithmique et un Neural Network.
 Les NN (Neural Network) permettent d'apprendre les relations non linéaires et les interactions complexes pouvant exister entre les variables explicatives. 

<br>

## 1. Présentation des données

Les données utilisées dans ce projet proviennent de Kaggle. Le dataset contient environ 1 340 observations, chacune correspondant à un assuré.
La variable cible est `claim`, qui représente le montant de sinistre associé à chaque observation, exprimé en USD.
##### Les variables explicatives sont :
| Variable | Description|
| ----------------- | ----------------- | 
| age | Âge de l'assuré| 
| gender | Genre | 
| bmi | Indice de masse corporelle | 
| bloodpressure | Pression artérielle | 
| diabetic | Statut diabétique | 
| children | Nombre d'enfants | 
| region | Région géorgraphique | 
| smoker | Statut fumeur | 

<br>

## 2. Analyse exploratoire

La distribution des montants présente une **asymétrie positive**, avec un coefficient de skewness d'environ **1,5**.

Les principales statistiques descriptives sont les suivantes :

| Indicateur | Valeur approximative |
|---|---:|
| Moyenne | 13 253 USD |
| Médiane | 9 370 USD |
| Minimum | 1 122 USD |
| Maximum | 63 770 USD |
| 95e percentile | 41 125 USD |

Environ **95 % des montants observés sont inférieurs ou égaux à 41 125 USD**, tandis que les **5 % les plus élevés représentent environ 18 % du montant total des sinistres**.
Il y a une forte concentration des coûts, pouvant avoir un impact financier important.

#### 2.1 Influence du statut fumeur

Les montants moyens observés sont approximativement :

- **Fumeurs : 32 050 USD**
- **Non-fumeurs : 8 420 USD**

Le statut fumeur apparaît fortement associé au montant des sinistres.

### 2.2 Relation entre BMI et le montant des sinistres
L'analyse exploratoire suggère que la relation entre l'indice de masse corporelle et le montant des sinistres  varie selon le statut fumeur.

Une association positive plus marquée a été observée chez les fumeurs, tandis que la relation apparaît plus faible chez les non-fumeurs.
Cette observation suggère une interaction potentielle entre bmi et smoker.

Elle motive l'étude de méthodes capables de représenter des relations non linéaires et des interactions complexes.


## 3. Préparation des données

### 3.1 Séparation des données

Les observations ont été réparties en trois ensembles :

| Ensemble | Proportion | Observations | Objectif |
|---|---|---|---|
| Entraînement | 70 % | 937 | Estimer les paramètres des modèles |
| Validation | 15 % | 201 | Comparer les modèles et ajuster les hyperparamètres |
| Test | 15 % | 202 | Évaluer les performances finales |

Cette séparation permet de distinguer l'apprentissage, la sélection des modèles et l'évaluation finale sur des observations d'autres observations.

### 3.2 Traitement des valeurs manquantes

5 valeurs manquantes ont été identifiées dans la variable `age`.

Elles ont été remplacées par la médiane de l'âge calculée sur l'ensemble du data train, puis cette même valeur a été utilisée pour les ensembles de validation et de test.

Cette méthode évite d'utiliser des informations provenant des jeux de validation et de test pendant la préparation du modèle.

### 3.3 Encodage des variables catégorielles

Les variables binaires (`smoker`, `diabetic` et `gender`) ont été encodées et la variable `region`, qui comporte quatre catégories, a été représentée par quatre colonnes indicatrices avec le one-hot encoding pour le NN.

### 3.4 Standardisation

Les variables numériques `age`, `bmi`, `bloodpressure` et `children` ont été standardisées.

La moyenne et l'écart-type ont été calculés exclusivement sur les données d'entraînement, puis appliqués à la validation et au test.


Pour l'entraînement final du NN, la variable cible `claim` a été **divisée par 1 000** afin de faciliter l'apprentissage numérique. Les prédictions ont ensuite été reconverties en USD avant le calcul des métriques.

## 4. Modèles étudiés

### 4.1 Régression linéaire multiple

La régression linéaire multiple est un modèle de référence simple et interprétable.
Elle permet d'estimer le montant des sinistres à partir d'une combinaison linéaire des variables explicatives et elle a pour avantage de fournir des coefficients directement interprétables.

Cependant, la spécification étudiée ne comporte pas de transformations non linéaires ni d'interactions ajoutées explicitement.

### 4.2 GLM Gamma avec lien logarithmique

Un modèle linéaire généralisé de famille Gamma et de lien logarithmique a également été étudié.
Puisque nous voulons faire la prédiction de `claim`, variable continue strictement positive et de distribution asymétrique.
Le lien logarithmique garantit une espérance conditionnelle prédite positive et permet une interprétation multiplicative de certains effets.

### 4.3 Neural Network

Le NN a été développé avec le package `keras3` sous R, en utilisant TensorFlow comme backend.

L'architecture finale comprend :

| Couche | Configuration |
|---|---|
| Entrée | 11 caractéristiques |
| Couche cachée | 16 neurones, activation ReLU |
| Sortie | 1 neurone, activation linéaire |

Le réseau contient **209 paramètres**.

L'activation ReLU permet d'introduire des non-linéarités, tandis que la sortie linéaire est adaptée à la prédiction d'une variable quantitative non bornée.

L'entraînement utilise :

- **Optimiseur :** Adam
- **Fonction de perte :** Mean Squared Error (MSE)
- **Métrique suivie :** Mean Absolute Error (MAE)
- **Batch size :** 32
- **Nombre maximal d'époques :** 150
- **Early stopping :** suivi de la MAE de validation, avec restauration des meilleurs poids

La couche de sortie linéaire peut produire des prédictions négatives. Cette propriété doit être prise en compte dans la perspective d'une application actuarielle où les montants de sinistres ne sont pas négatifs.

## 5. Évaluation des performances

Les modèles ont été évalués à l'aide de deux métriques complémentaires, la MAE qui mesure l'erreur absolue moyenne entre les montants observés et les montants prédits et la RMSE qui pénalise davantage les erreurs importantes.

## 6. Résultats


L'évaluation finale a été effectuée sur les 202 observations de test.

| Modèle | MAE (USD) | RMSE (USD) |
|---|---:|---:|
| GLM Gamma | 4 650,74 | 6 388,26 |
| Régression linéaire multiple | 4 773,37 | 6 422,40 |
| **Neural Network** | **4 086,43** | **5 515,05** |

Sur ce jeu de test, le réseau de neurones obtient les erreurs les plus faibles parmi les trois modèles comparés.

Par rapport au GLM Gamma, il présente une réduction d'environ :

- **12,1 % de la MAE**
- **13,7 % de la RMSE**

### 6.1 Interprétation

Les résultats montrent qu'une architecture de réseau de neurones relativement simple peut apporter un gain de précision dans la prédiction des montants individuels de sinistres.
Ce gain est compatible avec l'hypothèse selon laquelle une modélisation plus flexible permet de mieux représenter certaines relations complexes présentes dans les données.

Par ailleurs, les modèles de régression comparés ne comportent pas d'interactions explicites ni de transformations non linéaires supplémentaires. Une comparaison avec des GLM enrichis pourrait donc conduire à des résultats différents.

## 7. Analyse de l'interaction BMI × statut fumeur

Afin d'examiner les relations apprises par le réseau, une analyse de dépendance partielle (Partial Dependence Plot) a été réalisée.
Cette méthode consiste à faire varier le BMI et le statut fumeur tout en conservant les autres caractéristiques des observations, puis à calculer les prédictions moyennes du réseau.

On observe alors, chez les non-fumeurs, les montants prédits augmentent relativement peu avec le BMI.
Chez les fumeurs, les montants prédits augmentent beaucoup plus fortement avec le BMI.

Les courbes n'étant pas parallèles, cette analyse met en évidence une interaction entre BMI et statut fumeur dans les prédictions du réseau.

Le NN a appris à représenter une relation dont l'intensité dépend du profil de l'assuré.
Cependant, cette analyse porte sur le comportement prédictif du modèle et ne permet pas d'établir une relation causale.

## 8. Technologies utilisées

Le projet a été réalisé principalement avec les outils suivants :

| Outil | Utilisation |
|---|---|
| R | Analyse statistique et préparation des données |
| `rsample` | Séparation entraînement, validation et test |
| `stats::lm()` | Régression linéaire multiple |
| `stats::glm()` | GLM Gamma |
| `keras3` | Construction et entraînement du réseau de neurones |
| TensorFlow | Backend d'apprentissage |
| `reticulate` | Interface entre R et Python |


## Conclusion

Ce projet avait pour objectif de comparer plusieurs approches de modélisation pour la prédiction des montants de sinistres en assurance santé.
Les résultats obtenus montrent que le Neural Network développé présente de meilleures performances prédictives que la régression linéaire multiple et le GLM Gamma étudiés.

Avec une **MAE de 4 086 USD et une RMSE de 5 515 USD**, le réseau de neurones réduit respectivement ces erreurs d'environ **12,1 % et 13,7 % par rapport au GLM Gamma**.
Ces résultats illustrent l'intérêt potentiel des réseaux de neurones comme outils complémentaires aux approches traditionnelles en actuariat.

Ils montrent également qu'un modèle sophistiqué n'est pas une garantie de performance car la préparation des données, la méthode d'entraînement, l'évaluation et la comparaison avec des modèles de référence jouent un rôle essentiel.

### Source de données : https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health/discussion?sort=recent-comments
