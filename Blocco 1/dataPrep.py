import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sktime.datasets import load_arrow_head
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm


# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/data/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/'
series = []
for i in range (1,61):
    df = pd.read_csv(folder + str(i) + '_AHA_1sec.csv')
    # Calculating magnitude
    magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
    magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))
    # Concat
    series.append(pd.concat([magnitude_D, magnitude_ND], ignore_index = True))


# Create a DataFrame with a single column of Series objects
X = pd.DataFrame({'series': series})
print(X.shape)
print(type(X))
print(X)

k_means = TimeSeriesKMeans(
    n_clusters=2,  # Number of desired centers
    init_algorithm="random",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    metric="dtw",  # Distance metric to use
    averaging_method="mean",  # Averaging technique to use
    random_state=1,
)

k_means.set_tags(**{"X_inner_mtype": "numpy3D",
"multivariate": False,
        "unequal_length": True,
        "missing_values": False,
        "multithreading": False})

k_means.fit(X)
plot_cluster_algorithm(k_means, X, k_means.n_clusters)
'''''
fig, axs = plt.subplot(235)
for x in series:
    i = 0
    axs[0] = plt.plot(x)
    i += 1
plt.show()
'''''