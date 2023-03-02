import json
import hashlib
import re
import os
import datetime
import matplotlib
import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from sktime.base import BaseEstimator
from elaborate_magnitude import elaborate_magnitude
from train_regressor import train_regressor


# Funzione per prendere la media del primo quartile dei massimi valori di una lista
def first_quartile_of_max(my_list):
    sorted_list = sorted(my_list, reverse=True)
    quartile_length = len(sorted_list) // 2
    if quartile_length == 0:
        return sorted_list[0]
    first_quartile = sum(sorted_list[:quartile_length]) / quartile_length
    return first_quartile


if os.getlogin() == 'david':
    data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
elif os.getlogin() == 'giord':
    data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
else:
    data_folder = input("Enter the data folder path: ")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

best_estimators_df = pd.read_csv('best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

estimators_specs_list = []

#TODO: controllare che abbiano tutti la stessa window_size
estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'concat'].iloc[0])
estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'ai'].iloc[0])
estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'difference'].iloc[0])


estimators_list = []
model_params_concat = ''

for estimators_specs in estimators_specs_list:
    parent_dir = "Trained_models/" + estimators_specs['method'] + "/" + str(estimators_specs['window_size']) + "_seconds/" + estimators_specs['model_type'].split(".")[-1] + "/"
    
    # TODO: Al momento carica tutti i modelli che trova nelle varie gridsearch
    for subdir in os.listdir(parent_dir):
        # Check if the current item is a directory
        if os.path.isdir(os.path.join(parent_dir, subdir)):
            # Access the current subdirectory
            estimator = BaseEstimator().load_from_path(os.path.join(parent_dir, subdir) + '/best_estimator.zip')
            estimators_list.append(estimator)
            print('Loaded -> ', os.path.join(parent_dir, subdir) + '/best_estimator.zip')
            model_params_concat = model_params_concat + str(estimator.get_params())


metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx')

reg_path = 'Regressors/regressor_'+ (hashlib.sha256((model_params_concat).encode()).hexdigest()[:10])

if not(os.path.exists(reg_path)):
    #Dobbiamo allenare il regressore
    train_regressor(data_folder, metadata, estimators_list, estimators_specs_list[0]['window_size'], estimators_specs_list[0]['method'], reg_path)


#save_week_stats(best_concat, best_ai, best_difference, regressor,)
