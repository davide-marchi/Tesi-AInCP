import os
import numpy as np
import joblib as jl
from create_windows import create_windows
from save_model_stats import save_model_stats

from sktime.classification.base import BaseClassifier

import importlib


def train_model(data_folder, specs, model_folder):

    # Split the string into the module and class names
    module_name, class_name = specs["class_type"].rsplit(".", 1)
    model = getattr(importlib.import_module(module_name), class_name)()

    model.set_params(**specs['params'])

    series, y_AHA, y_MACS, y = create_windows(data_folder, specs['window_size'], specs['method'], specs['patients'])

    # Create a DataFrame with a single column of Series objects
    X = np.array(series)

    if issubclass(type(model), BaseClassifier):
        print(type(model)," is a CLASSIFIER !")
    else:
        print(type(model)," is a CLUSTERER !")
        #model.fit_predict(X)

    #y_pred = model.fit(X, y)
    os.makedirs(model_folder, exist_ok = True)
    model.save(model_folder + "trained_model")

    #save_model_stats(X, y_AHA, y_MACS, y_pred, model, specs["params"]['n_clusters'], model_folder)

    #print("TRAINING COMPLETED (Model ready)")