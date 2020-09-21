#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, 2:]  # on prends les deux dernieres.
"""les données : sepal lenght, sepal width, petal lenght, petal width"""
Y = iris.target



n_classes = len(np.unique(Y))


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5


def display_histogram():
    """ici, un histogramme est fait pour chaque attribut de l'ensemble des données"""
    colors = ['red', 'blue', 'lime'] #pour les différentes classes
    feats = iris.feature_names

    fig, axes = plt.subplots(nrows=2, ncols=2) #création de 4 sous graphes
    ax0, ax1, ax2, ax3 = axes.flatten()
    all_axs = [ax0,ax1,ax2,ax3]

    data_par_classe = np.array([iris.data[Y == i] for i in range(n_classes)]) # ici on trie les données en fonction de leur classe

    for i in range(len(feats)): #pour chaque feature

        specifique_feat_by_class=[] #le tableau contiendra tous les tableaux qui ont l'indice i (la feature)
                                    # dans pour pouvoir faire un subplot par feature
        for e in range(n_classes):
            specifique_feat_by_class.append([j[i] for j in data_par_classe[e]])


        all_axs[i].hist(specifique_feat_by_class,10,histtype='bar',density=True,color=colors)
        all_axs[i].set_title(feats[i])      
    fig.tight_layout()

    plt.show()


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
    La distance intra : le point le plus éloigné par rapport à la moyenne des points.
    La distance interclasses"""



#display_figure()

#display_figures_with_means()


display_histogram()
# %%
