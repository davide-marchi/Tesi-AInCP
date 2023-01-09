import pandas as pd
import joblib as jl
from utils import create_windows

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

######################################################################################
WINDOW_SIZE=900
N_CLUSTERS=6
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


series, y, total, taken, lost = create_windows(WINDOW_SIZE, folder)
        
print("Total = " + str(total) + " seconds")
print("Trained on = " + str(taken) +" seconds (" + str(taken/total*100) + " % of total)")
print("Cutted out = " + str(lost) +" seconds (" + str(lost/total*100) + " % of total)")


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