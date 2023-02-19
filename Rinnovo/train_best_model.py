import os
import numpy as np
import joblib as jl
from create_windows import create_windows
from save_model_stats import save_model_stats

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sktime.classification.base import BaseClassifier


import importlib

def train_best_model(data_folder, model_folder, model_type, model_params, patients, method, window_size):

    # Split the string into the module and class names
    module_name, class_name = model_type.rsplit(".", 1)
    model = getattr(importlib.import_module(module_name), class_name)()

    series, y_AHA, y_MACS, y = create_windows(data_folder, patients, method, window_size)

    # Create a DataFrame with a single column of Series objects
    X = np.array(series)
    y = np.array(y)

    if issubclass(type(model), BaseClassifier):

        print(type(model)," is a CLASSIFIER !")
        
        param_grid = model_params
        parameter_tuning_method = GridSearchCV(model, param_grid, cv=KFold(n_splits=5))
        # f1 score consigliato da Sirbu?

        parameter_tuning_method.fit(X, y)
        y_pred = parameter_tuning_method.predict(X)

    else:
        print(type(model)," is a CLUSTERER !")
        y_pred = model.fit_predict(X)

    os.makedirs(model_folder, exist_ok = True)
    model.save(model_folder + "model")
    print('model saved : ', model_type)

    #save_model_stats(X, y_AHA, y_MACS, y_pred, model, specs["params"]['n_clusters'], model_folder)

    #print("TRAINING COMPLETED (Model ready)")