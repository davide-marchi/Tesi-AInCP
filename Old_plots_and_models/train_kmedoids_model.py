import os
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from create_windows import create_windows
from save_model_stats import save_model_stats

from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

def train_kmedoids_model(folder, patients, WINDOW_SIZE, N_CLUSTERS, INIT_ALGORITHM, MAX_ITER, METRIC):

    modelname='KMEDOIDS_K'+str(N_CLUSTERS)+'_W'+str(WINDOW_SIZE)+'_I'+str(MAX_ITER)+'_'+INIT_ALGORITHM+'_'+METRIC
    
    series, y_AHA, y_MACS, total, taken, lost = create_windows(WINDOW_SIZE, folder, patients)

    # Create a DataFrame with a single column of Series objects
    X = pd.DataFrame({'series': series})

    k_medoids = TimeSeriesKMedoids(
        n_clusters=N_CLUSTERS,  # Number of desired centers
        init_algorithm=INIT_ALGORITHM,  # Center initialisation technique
        max_iter=MAX_ITER,  # Maximum number of iterations for refinement on training set
        metric=METRIC,  # Distance metric to use
        verbose=False
    )

    model_folder = "./Blocco 1/" + str(patients)+"_patients_kmedoids_"+METRIC+"/"

    if not os.path.exists(model_folder + modelname + "/" + modelname):
        y_pred = k_medoids.fit_predict(X)
        os.makedirs(model_folder + modelname)
        jl.dump(k_medoids, model_folder + modelname + "/" + modelname)
        with open(model_folder + modelname + '/parametri.txt', 'w') as f:
            f.write('n_clusters = ' + str(N_CLUSTERS) + '\n')  # Number of desired centers
            f.write('init_algorithm = ' + str(INIT_ALGORITHM) + '\n')  # Center initialisation technique
            f.write('max_iter = ' + str(MAX_ITER) + '\n')  # Maximum number of iterations for refinement on training set
            f.write('metric = ' + str(METRIC) + '\n')  # Distance metric to use

            f.write('Total = ' + str(total) + ' seconds\n')
            f.write('Trained on = ' + str(taken) + ' seconds (' + str(taken/total*100) + ' % of total)\n')
            f.write('Cutted out = ' + str(lost) + ' seconds (' + str(lost/total*100) + ' % of total)\n\n')

        save_model_stats(X, y_AHA, y_MACS, y_pred, k_medoids, modelname, N_CLUSTERS, model_folder)

    else:
        print("The model has already been trained!")
        k_medoids = jl.load(model_folder + modelname + "/" + modelname)

    #plot_cluster_algorithm(k_medoids, X, k_medoids.n_clusters)

    #print("TRAINING COMPLETED (Model " + modelname + " ready)")