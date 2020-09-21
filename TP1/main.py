#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, 2:]  # we only take the first two features.
Y = iris.target

n_classes = len(np.unique(Y))


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

def display_figure():
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

def getMean():
        moyenne_class = np.array([X[Y == i].mean(axis=0)
                                    for i in range(n_classes)])
        print(moyenne_class)
        return moyenne_class

        
def display_figures_with_means():
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    moyennes = getMean()
    moyennes_x = [i[0] for i in moyennes]
    moyennes_y = [i[1] for i in moyennes]
    plt.scatter(moyennes_x,moyennes_y,s=90)

    plt.show()

def get_informations():
    """renvoie toutes les informations nécéssaires à la partie 1
    pour montrer qu'elles sont toutes séparées, on va afficher dans un premier temps
    la moyenne des éléments pour chaque clasese;
    la distance  entre le centre d'une classe et l'élémenent le plus loin"""



#display_figure()

#display_figures_with_means()


get_informations()