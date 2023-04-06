import os
import json
import importlib
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sktime.classification.base import BaseClassifier
from sklearn.metrics import f1_score
from create_windows import create_windows


def scorer_f(estimator, X_train, Y_train):
    y_pred = estimator.predict(X_train)
    if issubclass(type(estimator), BaseClassifier):
        return f1_score(Y_train, y_pred)
    else:
        inverted_y_pred = [1 if item == 0 else 0 for item in y_pred]
        return max(f1_score(Y_train, y_pred),f1_score(Y_train, inverted_y_pred))


def train_best_model(data_folder, model_folder, model_type, model_params, method, window_size):

    # Split the string into the module and class names
    module_name, class_name = model_type.rsplit(".", 1)
    model = getattr(importlib.import_module(module_name), class_name)()

    X, _, _, y = create_windows(data_folder, method, window_size)
 
    param_grid = model_params
    #                                                             dobbiamo fixare il seed?
    parameter_tuning_method = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), n_jobs=-1, return_train_score=True, verbose=3, scoring=scorer_f)
    parameter_tuning_method.fit(X, y)

    model = parameter_tuning_method.best_estimator_
    

    stats_folder = model_folder + 'training_stats/'
    os.makedirs(stats_folder, exist_ok = True)
    model.save(model_folder + "model")
    print('model saved : ', model_type)

    pd.DataFrame(parameter_tuning_method.cv_results_).to_csv(stats_folder + "cv_results.csv")
    with open(stats_folder + 'params.json', 'w') as f:
        f.write(json.dumps(parameter_tuning_method.best_params_, indent=4))