L’objectif de ce projet consiste à étudier une banque de données sur la consommation des
voitures. 
Nous avons analysé les données sur la consommation des voitures et prédire la consommaation en miles par gallon (mpg)
en utilisant une arbre de regression de type CART et par la suite une foret aléatoire.



EXERCICE 1: Regression 

1. Importation des données
Le fichier "auto-mpg.csv" est utilisé comme jeu de données. Elle est disponible sur 
le dépot de l’UCI (https://archive.ics.uci.edu/ml/datasets/Auto+MPG)

1.2. Analyse des données
Nous avons utiliser des librairies telles que pandas,numpy, matplotlib, seaborn, plotly 
pour nous aider à calculer les statistiques descriptives sont calculées.

Des liens comme https://seaborn.pydata.org/tutorial.html pour seaborn et 
https://plotly.com/python/ pour plotly nous ont aidé.

Les données manquantes ont été supprimé.
Les variables sont analysées individuellement.

2. Arbre de Régression
-Définition des données d’apprentissage et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y ,random_state=22, test_size =0.2)

-Modèle d'Arbre de Régression de type CART
Ce lien a été utile pour comprendre comment contruire un arbre de regression et commprendre ces paramètres
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html 

-Pour la visualisation de l’arbre, nous l'avons fait sous deux forme une avec la fonction 
export_text (from sklearn.tree import export_text)
et installer graphviz par la commande conda install python-graphviz

-Imoortance des variables avec clf.feature_importances_ 

-Nous avons par la suite calculer les mesures d'erreurs (MSE, MAE, R2) de sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 


3-Forêt Aléatoire avec la fonction RandomForestClassifier()

Nous avons optmisés les hyperparamètres avec une grille de recherche aléatoire
from sklearn.model_selection import RandomizedSearchCV

Par la suite réaliser la meme tache que l'arbre de regression
Les variables importantes sont affichées.


Partie 2: Clustering

1- Projection en 2D avec l'Analyse en Composantes Principales (PCA)
La bibliothèque Yellowbrick a été utilisée pour la projection en 2D avec PCA
et a permis de visualiser les données dans un espace réduit.

Nous avons utilisé ce code tiré de https://www.scikit-yb.org/en/latest/api/features/pca.html
from yellowbrick.datasets import load_credit
from yellowbrick.features import PCA
visualizer = PCA(scale=True, proj_features=False)
visualizer.fit_transform(X, y)
visualizer.show()


3. Choix et application de la méthode de clustering
Nous avons choisit la méthode K-means pour le clustering.
Sa documentation est disponible sur https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
La méthode du coude (Elbow method) est utilisée pour déterminer le nombre optimal de clusters.

4. Visualisation des clusters en 2D
Le code a été inspiré de https://www.labri.fr/perso/zemmari/l2-ai/html/l2_lab1_correction.html

5. Métriques d'évaluation
La silhouette est calculée comme métrique d'évaluation.
Notre code est tirée de sa documentation disponible sur sklearn via
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html