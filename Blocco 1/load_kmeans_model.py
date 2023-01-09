import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from utils import create_windows
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')


series, y, total, taken, lost = create_windows(900, folder)

print("taken = " + str(taken) + "  lost = " + str(lost) + "  on a total of = "+str(total)+"  percent lost = " + str((lost/total)*100) +"%")

# Create a DataFrame with a single column of Series objects
X = pd.DataFrame({'series': series})

stats = pd.DataFrame(columns=['clusters', 'AHA'])

k_means = jl.load('KMEANS_K4_W900_I10_kmeans++_dtw_dba')
y_pred = k_means.predict(X)
print(y_pred)
print(y)

stats['clusters'] = y_pred
stats['AHA'] = y
stats = stats.groupby('clusters').agg(list)
print(stats)

for i in range(0,stats.shape[0]):
    print(stats['clusters'].iloc[i])
    s = pd.Series(stats['AHA'].iloc[i])
    s.describe()

print('score: ', k_means.score(X))
plt.scatter(y, y_pred)
plt.show
plot_cluster_algorithm(k_means, X, k_means.n_clusters)
