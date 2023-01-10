import os
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from utils import create_windows
from utils import save_model_stats

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

######################################################################################
WINDOW_SIZE=900
N_CLUSTERS=3
INIT_ALGORITHM='random'
MAX_ITER=3
METRIC="euclidean"
AVERAGING_METHOD="mean"
#######################################################################################

modelname='KMEANS_K'+str(N_CLUSTERS)+'_W'+str(WINDOW_SIZE)+'_I'+str(MAX_ITER)+'_'+INIT_ALGORITHM+'_'+METRIC+'_'+AVERAGING_METHOD

# Creating dataframe
#folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'


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
    

if not os.path.exists("./Blocco 1/" + modelname + "/" + modelname):
    y_pred = k_means.fit_predict(X)
    os.makedirs("./Blocco 1/" + modelname)
    jl.dump(k_means, "./Blocco 1/" + modelname + "/" + modelname)
    with open('./Blocco 1/' + modelname + '/parametri.txt', 'w') as f:
        f.write('n_clusters = ' + str(N_CLUSTERS) + '\n')  # Number of desired centers
        f.write('init_algorithm = ' + str(INIT_ALGORITHM) + '\n')  # Center initialisation technique
        f.write('max_iter = ' + str(MAX_ITER) + '\n')  # Maximum number of iterations for refinement on training set
        f.write('metric = ' + str(METRIC) + '\n')  # Distance metric to use
        f.write('averaging_method = ' + str(AVERAGING_METHOD))  # Averaging technique to use

    save_model_stats(X, y, y_pred, k_means, modelname)

else:
    print("The model has already been trained!")
    k_means = jl.load("./Blocco 1/" + modelname + "/" + modelname)

#plot_cluster_algorithm(k_means, X, k_means.n_clusters)