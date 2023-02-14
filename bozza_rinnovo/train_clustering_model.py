import os
import numpy as np
import joblib as jl
from create_windows import create_windows
from save_model_stats import save_model_stats

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.k_shapes import TimeSeriesKShapes
from sktime.clustering.kernel_k_means import TimeSeriesKernelKMeans
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm


def train_clustering_model(folder, plot_clusters, operation_type, patients, model_type, WINDOW_SIZE, params):

    if (model_type == 'KMeans'):
        AVERAGING_METHOD = 'mean' # defaults
        if params['metric'] == 'dtw':
            AVERAGING_METHOD = 'dba'
        model_name='KMEANS_K'+str(params['n_clusters'])+'_W'+str(WINDOW_SIZE)+'_'+params['init_algorithm']+'_'+params['metric']+'_'+AVERAGING_METHOD
        model = TimeSeriesKMeans()
        model.set_params(**params)
    elif(model_type == 'KMedoids'):
        model_name='KMEDOIDS_K'+str(params['n_clusters'])+'_W'+str(WINDOW_SIZE)+'_'+params['init_algorithm']+'_'+params['metric']
        model = TimeSeriesKMedoids()
        model.set_params(**params)
    elif(model_type == 'KShapes'):
        model_name='KSHAPES_K'+str(params['n_clusters'])+'_W'+str(WINDOW_SIZE)
        model = TimeSeriesKShapes()
        model.set_params(**params)
    elif(model_type == 'KernelKMeans'):
        model_name='KERNELKMEANS_K'+str(params['n_clusters'])+'_W'+str(WINDOW_SIZE)
        model = TimeSeriesKernelKMeans()
        model.set_params(**params)
    else:
        print("Invalid model requested")
        exit(1)

    series, y_AHA, y_MACS, total, taken, lost = create_windows(WINDOW_SIZE, folder,operation_type, patients)

    # Create a DataFrame with a single column of Series objects
    X = np.array(series)

    model_folder = "./Blocco 1/"+operation_type+"_version/" + str(patients) + "_patients/" + model_type + "/" + model_name

    if not os.path.exists(model_folder + "/trained_model"):
        y_pred = model.fit_predict(X)
        os.makedirs(model_folder)
        jl.dump(model, model_folder + "/trained_model")
        with open(model_folder + '/parametri.txt', 'w') as f:
            f.write('Total = ' + str(total) + ' seconds\n')
            f.write('Trained on = ' + str(taken) + ' seconds (' + str(taken/total*100) + ' % of total)\n')
            f.write('Cutted out = ' + str(lost) + ' seconds (' + str(lost/total*100) + ' % of total)\n\n')

        save_model_stats(X, y_AHA, y_MACS, y_pred, model, params['n_clusters'], model_folder)

    else:
        print("The model has already been trained!")
        model = jl.load(model_folder + "/trained_model")

    if plot_clusters:
        plot_cluster_algorithm(model, X, model.params['n_clusters'])

    #print("TRAINING COMPLETED (Model " + model_name + " ready)")