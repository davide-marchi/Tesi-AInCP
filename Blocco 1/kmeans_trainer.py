import pandas as pd
import numpy as np
import joblib as jl
import math

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

######################################################################################
WINDOW_SIZE=300
N_CLUSTERS=3
INIT_ALGORITHM='kmeans++'
MAX_ITER=10
METRIC="dtw"
AVERAGING_METHOD="dba"
#######################################################################################

modelname='KMEANS_K'+str(N_CLUSTERS)+'_W'+str(WINDOW_SIZE)+'_I'+str(MAX_ITER)+'_'+INIT_ALGORITHM+'_'+METRIC+'_'+AVERAGING_METHOD

# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
#metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')


series = []
#y = []
lost = 0
total = 0
taken = 0
to_discard = 0
for j in range (1,61):
    df = pd.read_csv(folder + 'data/' + str(j) + '_AHA_1sec.csv')
    #assumiamo che dfshape[0]>=WINDOW_SIZE
    scart = (df.shape[0] % WINDOW_SIZE)/2
    total += df.shape[0]
    df_cut = df.iloc[math.ceil(scart):df.shape[0]-math.floor(scart)]
    lost += df.shape[0]-df_cut.shape[0]
    # Calculating magnitude
    magnitude_D = np.sqrt(np.square(df_cut['x_D']) + np.square(df_cut['y_D']) + np.square(df_cut['z_D']))
    magnitude_ND = np.sqrt(np.square(df_cut['x_ND']) + np.square(df_cut['y_ND']) + np.square(df_cut['z_ND']))
    for i in range (0, len(magnitude_D), WINDOW_SIZE):
        chunk_D = magnitude_D.iloc[i:i + WINDOW_SIZE]
        chunk_ND = magnitude_ND.iloc[i:i + WINDOW_SIZE]
        # Concat
        magnitude_concat = pd.concat([chunk_D, chunk_ND], ignore_index = True)
        series.append(magnitude_concat)
        taken += len(magnitude_concat)/2
        
print("taken = " + str(taken) + "  lost = " + str(lost) + "  on a total of = "+str(total)+"  percent lost = " + str((lost/total)*100) +"%")

# Create a DataFrame with a single column of Series objects
X = pd.DataFrame({'series': series})

k_means = TimeSeriesKMeans(
    n_clusters=N_CLUSTERS,  # Number of desired centers
    init_algorithm=INIT_ALGORITHM,  # Center initialisation technique
    max_iter=MAX_ITER,  # Maximum number of iterations for refinement on training set
    metric=METRIC,  # Distance metric to use
    averaging_method=AVERAGING_METHOD,  # Averaging technique to use
    verbose=True
)
    

y_pred = k_means.fit_predict(X)
jl.dump(k_means, modelname)
plot_cluster_algorithm(k_means, X, k_means.n_clusters)
print(y_pred)