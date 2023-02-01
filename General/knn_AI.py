"""
================================
Nearest Neighbors Classification
================================

Sample usage of Nearest Neighbors classification.
It will plot the decision boundaries for each class.

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay

n_neighbors = 10
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'


df = pd.read_excel(folder+'metadata2022_04.xlsx')

X = df[['AI_aha','AI_week']]
y = df['MACS']

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue","red"])
cmap_bold = ["darkorange", "c", "darkblue","darkred"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel='AI_AHA',
        ylabel='AI_week',
        shading="auto",
    )

    # Plot also the training points
    sns.scatterplot(
        x=df['AI_aha'],
        y=df['AI_week'],
        hue=df['MACS'],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.title(
        "4-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

plt.show()