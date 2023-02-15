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

import importlib


def train_clustering_model(folder, plot_clusters, model, model_name):

    '''
    if (model['name'] == 'KMeans'):
        AVERAGING_METHOD = 'mean' # defaults
        if params['metric'] == 'dtw':
            AVERAGING_METHOD = 'dba'
        model_name='KMEANS_K'+str(params['n_clusters'])+'_W'+str(model['window_size'])+'_'+params['init_algorithm']+'_'+params['metric']+'_'+AVERAGING_METHOD
        model = TimeSeriesKMeans()
        model.set_params(**params)
    elif(model['name'] == 'KMedoids'):
        model_name='KMEDOIDS_K'+str(params['n_clusters'])+'_W'+str(model['window_size'])+'_'+params['init_algorithm']+'_'+params['metric']
        model = TimeSeriesKMedoids()
        model.set_params(**params)
    elif(model['name'] == 'KShapes'):
        model_name='KSHAPES_K'+str(params['n_clusters'])+'_W'+str(model['window_size'])
        model = TimeSeriesKShapes()
        model.set_params(**params)
    elif(model['name'] == 'KernelKMeans'):
        model_name='KERNELKMEANS_K'+str(params['n_clusters'])+'_W'+str(model['window_size'])
        model = TimeSeriesKernelKMeans()
        model.set_params(**params)
    else:
        print("Invalid model requested")
        exit(1)
    '''
    class_string = model["class_type"]

    # Split the string into the module and class names
    module_name, class_name = class_string.rsplit(".", 1)

    # Import the module
    module = importlib.import_module(module_name)

    # Get the class object from the module
    class_obj = getattr(module, class_name)

    modello = class_obj()

    modello.set_params(**model['params'])

    series, y_AHA, y_MACS, total, taken, lost = create_windows(model['window_size'], folder,model['method'], model['patients'])

    # Create a DataFrame with a single column of Series objects
    X = np.array(series)

    model_folder = "./Blocco 1/"+model['method']+"_version/" + str(model['patients']) + "_patients/" + model['name'] + "/" + model_name

    if not os.path.exists(model_folder + "/trained_model"):
        y_pred = modello.fit_predict(X)
        os.makedirs(model_folder)
        jl.dump(modello, model_folder + "/trained_model")
        with open(model_folder + '/parametri.txt', 'w') as f:
            f.write('Total = ' + str(total) + ' seconds\n')
            f.write('Trained on = ' + str(taken) + ' seconds (' + str(taken/total*100) + ' % of total)\n')
            f.write('Cutted out = ' + str(lost) + ' seconds (' + str(lost/total*100) + ' % of total)\n\n')

        save_model_stats(X, y_AHA, y_MACS, y_pred, modello, model["params"]['n_clusters'], model_folder)

    else:
        print("The model has already been trained!")
        modello = jl.load(model_folder + "/trained_model")

    if plot_clusters:
        plot_cluster_algorithm(modello, X, modello.n_clusters)

    print("TRAINING COMPLETED (Model " + model_name + " ready)")