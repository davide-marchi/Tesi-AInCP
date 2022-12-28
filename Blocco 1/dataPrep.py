import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm


# Creating dataframe
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/'
df = pd.read_csv(folder + 'only AC/data/9_AHA_1sec.csv')

# Calculating magnitude
magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

# Concat
mag_concat = pd.concat([magnitude_D, magnitude_ND], ignore_index = True)

print(mag_concat.shape)
print(type(mag_concat))
print(mag_concat)


# Create some Series objects
s1 = pd.Series([1, 2, 3])
s2 = pd.Series([4, 5, 6])
s3 = pd.Series([7, 8, 9])

# Create a DataFrame with a single column of Series objects
X = pd.DataFrame({'column': [s1, s2, s3]})
print(X.shape)
print(type(X))
print(X)

k_means = TimeSeriesKMeans(
    n_clusters=5,  # Number of desired centers
    init_algorithm="forgy",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    metric="dtw",  # Distance metric to use
    averaging_method="mean",  # Averaging technique to use
    random_state=1,
)

k_means.fit(X)
plot_cluster_algorithm(k_means, X, k_means.n_clusters)

# Polotting the concatted data
plt.plot(mag_concat)
plt.show()