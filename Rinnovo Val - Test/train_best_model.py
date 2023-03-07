import os
import json
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sktime.classification.base import BaseClassifier
from sklearn.metrics import f1_score
from create_windows import create_windows


def scorer_f(estimator, X_train, Y_train):
    y_pred = estimator.predict(X_train)
    if issubclass(type(estimator), BaseClassifier):
        return f1_score(Y_train, y_pred, average='weighted')
    else:
        inverted_y_pred = [1 if item == 0 else 0 for item in y_pred]
        return max(f1_score(Y_train, y_pred, average='weighted'),f1_score(Y_train, inverted_y_pred, average='weighted'))


def train_best_model(data_folder, gridsearch_folder, model_type, model_params, method, window_size):

    # Split the string into the module and class names
    module_name, class_name = model_type.rsplit(".", 1)
    model = getattr(importlib.import_module(module_name), class_name)()

    X, y_AHA, _, y = create_windows(data_folder, method, window_size)
 
    param_grid = model_params
    #                                                             dobbiamo fixare il seed?
    parameter_tuning_method = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), n_jobs=5, return_train_score=True, verbose=3, scoring=scorer_f)
    parameter_tuning_method.fit(X, y)

    estimator = parameter_tuning_method.best_estimator_

    if issubclass(type(estimator), BaseClassifier):
        hemi_cluster = 1
    else:    
        y_pred = estimator.predict(X)
        hemi_cluster = 1 if np.mean([x for x, y in zip(y_AHA, y_pred) if y == 0]) > np.mean([x for x, y in zip(y_AHA, y_pred) if y == 1]) else 0

    stats_folder = gridsearch_folder + 'GridSearchCV_stats/'
    os.makedirs(stats_folder, exist_ok = True)
    pd.DataFrame(parameter_tuning_method.cv_results_).to_csv(stats_folder + "cv_results.csv")
    
    with open(stats_folder + 'best_estimator_stats.json', 'w') as f:
        f.write(json.dumps({"Best index":int(parameter_tuning_method.best_index_), "Best score":parameter_tuning_method.best_score_, "Refit time":parameter_tuning_method.refit_time_, "Best params": parameter_tuning_method.best_params_, "Hemi cluster": hemi_cluster}, indent=4))

    estimator.save(gridsearch_folder + "best_estimator")
    print('Best estimator saved\n\n------------------------------------------------\n')