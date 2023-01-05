import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sktime.datasets import load_arrow_head
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

from sklearn.metrics import silhouette_score



# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')


series = []
y = []
lost = 0
total = 0
taken = 0
for j in range (1,61):
    df = pd.read_csv(folder + 'data/' + str(j) + '_AHA_1sec.csv')
    # Calculating magnitude
    magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
    magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))
    for i in range (0, len(magnitude_D), 300):
        chunk_D = magnitude_D[i:i + 300]
        chunk_ND = magnitude_ND[i:i + 300]
        # Concat
        magnitude_concat = pd.concat([chunk_D, chunk_ND], ignore_index = True)
        if len(magnitude_concat) == 600:
            series.append(round(magnitude_concat))
            y.append(metadata['AHA'].iloc[j-1])
            taken += len(magnitude_concat)/120
            
        else: 
            print(len(magnitude_concat)/120)
            lost += len(magnitude_concat)/120

        total += len(magnitude_concat)/120

print("taken = " + str(taken) + "  lost = " + str(lost) + "  on a total of = "+str(total)+"  percent lost = " + str((lost/total)*100) +"%")

# Create a DataFrame with a single column of Series objects
X = pd.DataFrame({'series': series})

k_means = TimeSeriesKMeans()

k_means.load_from_path(folder + 'kmeansDBA.zip')
print(k_means.is_fitted)

#plot_cluster_algorithm(k_means, X, k_means.n_clusters)
