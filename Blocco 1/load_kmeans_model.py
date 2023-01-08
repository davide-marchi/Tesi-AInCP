import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from utils import create_windows
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')


series, y, total, taken, lost = create_windows(300, folder)

print("taken = " + str(taken) + "  lost = " + str(lost) + "  on a total of = "+str(total)+"  percent lost = " + str((lost/total)*100) +"%")

# Create a DataFrame with a single column of Series objects
X = pd.DataFrame({'series': series})

stats = pd.DataFrame()

k_means = jl.load('KMEANS_K4_W300_I10_kmeans++_dtw_dba')
y_pred = k_means.predict(X)
print(y_pred)
print(y)

stats['clusters'] = y_pred
stats['AHA'] = y
stats = stats.groupby('clusters').agg(list)
print(stats)
#print(stats.loc["clusters",1])
#s = pd.Series(stats.loc["AHA",1])
#s.describe()
print('score: ', k_means.score(X))
plt.scatter(y, y_pred)
plt.show
plot_cluster_algorithm(k_means, X, k_means.n_clusters)
