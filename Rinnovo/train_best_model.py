import os
import numpy as np
import joblib as jl
from create_windows import create_windows
from save_model_stats import save_model_stats

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sktime.classification.base import BaseClassifier

import importlib

def train_best_model(data_folder, model_folder, model_type, model_params, method, window_size):

    # Split the string into the module and class names
    module_name, class_name = model_type.rsplit(".", 1)
    model = getattr(importlib.import_module(module_name), class_name)()

    X, y_AHA, y_MACS, y = create_windows(data_folder, method, window_size)

    if issubclass(type(model), BaseClassifier):

        print(type(model)," is a CLASSIFIER !")
        
        param_grid = model_params
        parameter_tuning_method = GridSearchCV(model, param_grid, cv=KFold(n_splits=5), return_train_score=True, verbose=3)
        # f1 score consigliato da Sirbu?

        '''
        print("Pre")
        print(type(X_frag))
        print(X_frag.shape)
        print(type(y_frag))
        print(y_frag.shape)

        #X = np.array([[0, 1, 2, 3, 5, 6, 8], [0, 1, 2, 3, 5, 6, 9], [0, 1, 2, 3, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7]])
        #y = np.array([0, 0, 1, 1, 1])

        X = np.copy(X_frag)
        y = np.copy(y_frag)


        print("Post")
        print(type(X))
        print(X.shape)
        print(type(y))
        print(y.shape)
        '''

        parameter_tuning_method.fit(X, y)

        #print(parameter_tuning_method.best_score_)
        print(parameter_tuning_method.cv_results_)

        #y_pred = parameter_tuning_method.predict(X)

    else:
        print(type(model)," is a CLUSTERER !")
        #y_pred = model.fit_predict(X)

        param_grid = model_params
        parameter_tuning_method = GridSearchCV(model, param_grid, cv=KFold(n_splits=5), return_train_score=True, verbose=3)
        # f1 score consigliato da Sirbu?

        parameter_tuning_method.fit(X, y)
        print(parameter_tuning_method.cv_results_)


    os.makedirs(model_folder, exist_ok = True)
    model.save(model_folder + "model")
    print('model saved : ', model_type)

    #save_model_stats(X, y_AHA, y_MACS, y_pred, model, specs["params"]['n_clusters'], model_folder)

    #print("TRAINING COMPLETED (Model ready)")