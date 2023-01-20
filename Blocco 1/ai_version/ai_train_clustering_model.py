import os
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from ai_create_windows import ai_create_windows
from ai_save_model_stats import ai_save_model_stats

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.k_shapes import TimeSeriesKShapes
from sktime.clustering.kernel_k_means import TimeSeriesKernelKMeans
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm


def ai_train_clustering_model(folder, plot_clusters, patients, model_type, WINDOW_SIZE, N_CLUSTERS, INIT_ALGORITHM, METRIC):

    if (model_type == 'KMeans'):
        AVERAGING_METHOD = 'mean' # defaults
        if METRIC == 'dtw':
            AVERAGING_METHOD = 'dba'
        model_name='KMEANS_K'+str(N_CLUSTERS)+'_W'+str(WINDOW_SIZE)+'_'+INIT_ALGORITHM+'_'+METRIC+'_'+AVERAGING_METHOD
        model = TimeSeriesKMeans(
            n_clusters=N_CLUSTERS,  # Number of desired centers
            init_algorithm=INIT_ALGORITHM,  # Center initialisation technique
            metric=METRIC,  # Distance metric to use
            averaging_method=AVERAGING_METHOD,  # Averaging technique to use
            verbose=False
        )
    elif(model_type == 'KMedoids'):
        model_name='KMEDOIDS_K'+str(N_CLUSTERS)+'_W'+str(WINDOW_SIZE)+'_'+INIT_ALGORITHM+'_'+METRIC
        model = TimeSeriesKMedoids(
            n_clusters=N_CLUSTERS,  # Number of desired centers
            init_algorithm=INIT_ALGORITHM,  # Center initialisation technique
            metric=METRIC,  # Distance metric to use
            verbose=False
        )
    elif(model_type == 'KShapes'):
        model_name='KSHAPES_K'+str(N_CLUSTERS)+'_W'+str(WINDOW_SIZE)
        model = TimeSeriesKShapes(
            n_clusters=N_CLUSTERS,  # Number of desired centers
            verbose=False
        )
    elif(model_type == 'KernelKMeans'):
        model_name='KERNELKMEANS_K'+str(N_CLUSTERS)+'_W'+str(WINDOW_SIZE)
        model = TimeSeriesKernelKMeans(
            n_clusters=N_CLUSTERS,  # Number of desired centers
            verbose=False
        )
    else:
        print("Invalid model requested")
        exit(1)

    series, y_AHA, y_MACS, total, taken, lost = ai_create_windows(WINDOW_SIZE, folder, patients)

    # Create a DataFrame with a single column of Series objects
    X = pd.DataFrame({'series': series})

    model_folder = "./Blocco 1/ai_version/" + str(patients) + "_patients/" + model_type + "/" + model_name

    if not os.path.exists(model_folder + "/trained_model"):
        y_pred = model.fit_predict(X)
        os.makedirs(model_folder)
        jl.dump(model, model_folder + "/trained_model")
        with open(model_folder + '/parametri.txt', 'w') as f:
            f.write('Total = ' + str(total) + ' seconds\n')
            f.write('Trained on = ' + str(taken) + ' seconds (' + str(taken/total*100) + ' % of total)\n')
            f.write('Cutted out = ' + str(lost) + ' seconds (' + str(lost/total*100) + ' % of total)\n\n')

        ai_save_model_stats(X, y_AHA, y_MACS, y_pred, model, model_name, N_CLUSTERS, model_folder)

    else:
        print("The model has already been trained!")
        model = jl.load(model_folder + "/trained_model")

    if plot_clusters:
        plot_cluster_algorithm(model, X, model.n_clusters)

    #print("TRAINING COMPLETED (Model " + model_name + " ready)")