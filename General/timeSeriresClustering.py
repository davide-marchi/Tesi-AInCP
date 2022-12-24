from sklearn.cluster import KMeans
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df1 = np.array([[1,1], [2,2], [3,3], [4,4], [5,6]])
df2 = np.array([[1,1], [2,2], [3,3], [4,4], [5,15]])
df3 = np.array([[1,1], [2,2], [3,3], [4,4], [5,3]])
df4 = np.array([[1,1], [2,2], [3,3], [4,4], [5,2]])
df5 = np.array([[1,1], [2,2], [3,3], [4,4], [5,1]])



# Load the time series data
time_series_data = [df1, df2, df3, df4, df5]

# Define the number of clusters
n_clusters = 4

# Initialize the KMeans model
kmeans = KMeans(n_clusters=n_clusters)

# Compute the DTW distance between each pair of time series
dtw_distances = []
for i in range(len(time_series_data)):
  for j in range(i+1, len(time_series_data)):
    distance, _ = fastdtw(time_series_data[i], time_series_data[j])
    dtw_distances.append(distance)

# Fit the KMeans model using the DTW distances as input
kmeans.fit(dtw_distances)

# Predict the cluster labels for each time series
cluster_labels = kmeans.predict(dtw_distances)

# Assign the cluster labels to the time series data
time_series_data['cluster'] = cluster_labels


# Plot each time series separately
for i, time_series in time_series_data.groupby('cluster'):
  plt.plot(time_series, label=f'Cluster {i}')

# Add a legend and show the plot
plt.legend()
plt.show()
