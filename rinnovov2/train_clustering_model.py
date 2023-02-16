import os
import numpy as np
import joblib as jl
from create_windows import create_windows
from save_model_stats import save_model_stats

from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

import importlib


def train_clustering_model(data_folder, specs, model_folder, plot_clusters):

    # Split the string into the module and class names
    module_name, class_name = specs["class_type"].rsplit(".", 1)
    model = getattr(importlib.import_module(module_name), class_name)()

    model.set_params(**specs['params'])

    series, y_AHA, y_MACS, total, taken, lost = create_windows(specs['window_size'], data_folder, specs['method'], specs['patients'])

    # Create a DataFrame with a single column of Series objects
    X = np.array(series)

    y_pred = model.fit_predict(X)
    os.makedirs(model_folder)
    model.save(model_folder + "/trained_model")

    with open(model_folder + '/parametri.txt', 'w') as f:
        f.write('Total = ' + str(total) + ' seconds\n')
        f.write('Trained on = ' + str(taken) + ' seconds (' + str(taken/total*100) + ' % of total)\n')
        f.write('Cutted out = ' + str(lost) + ' seconds (' + str(lost/total*100) + ' % of total)\n\n')

    save_model_stats(X, y_AHA, y_MACS, y_pred, model, specs["params"]['n_clusters'], model_folder)

    if plot_clusters:
        plot_cluster_algorithm(model, X, model.n_clusters)

    print("TRAINING COMPLETED (Model ready)")