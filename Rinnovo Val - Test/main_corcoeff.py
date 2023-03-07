import os
import hashlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.base import BaseEstimator
from sklearn.metrics import r2_score
from predict_samples import predict_samples

if os.getlogin() == 'david':
    data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
elif os.getlogin() == 'giord':
    data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
else:
    data_folder = input("Enter the data folder path: ")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

best_estimators_df = pd.read_csv('best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx')
metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

estimators_specs_list = [row for index, row in best_estimators_df[best_estimators_df['mean_test_score'] >= 0.95].iterrows()]

estimators_list = []
model_params_concat = ''

for estimators_specs in estimators_specs_list:
    parent_dir = "Trained_models/" + estimators_specs['method'] + "/" + str(estimators_specs['window_size']) + "_seconds/" + estimators_specs['model_type'].split(".")[-1] + "/"
    
    for subdir in os.listdir(parent_dir):
        # Check if the current item is a directory
        if os.path.isdir(os.path.join(parent_dir, subdir)):

            # Access the current subdirectory
            with open(os.path.join(parent_dir, subdir) + '/GridSearchCV_stats/best_estimator_stats.json', "r") as stats_f:
                grid_search_best_params = json.load(stats_f)

            if str(grid_search_best_params['Best params']) == estimators_specs['params']:
                estimator = BaseEstimator().load_from_path(os.path.join(parent_dir, subdir) + '/best_estimator.zip')
                estimators_list.append({'estimator': estimator, 'method': estimators_specs['method'], 'window_size': estimators_specs['window_size']})
                print('Loaded -> ', os.path.join(parent_dir, subdir) + '/best_estimator.zip')
                model_params_concat = model_params_concat + str(estimator.get_params())

reg_path = 'Regressors/regressor_'+ (hashlib.sha256((model_params_concat).encode()).hexdigest()[:10])

hp_tot_list_list = []
y = []

for i in range (1, metadata.shape[0]+1):
    predictions, hp_tot_list, magnitude_D, magnitude_ND = predict_samples(data_folder, metadata, estimators_list, i)
    hp_tot_list_list.append(hp_tot_list)
    y.append(metadata['AHA'].iloc[i-1])

    #   hp_tot_list_list =                 y =
    #   [[ 95.0, 90.0, 80.0],              [56,
    #    [ 95.0, 90.0, 80.0],               70,
    #    [ 95.0, 90.0, 80.0],               80,
    #    [ 95.0, 90.0, 80.0]]               34]

estimators_number_list = []
corrcoef_list = []
r2_score_list = []

for i in range(len(estimators_list)):

    print('Taking ', str(i+1), ' estimators')

    X = [[sublist[:i]] for sublist in hp_tot_list_list]

    estimators_number_list.append(i+1)

    corrcoef_list.append(np.corrcoef(X, y))
    print(' - corrcoef:')
    print(np.corrcoef(X, y))

    y_pred = np.array(X).mean(axis=1)  # Predicted Y values are the means of each sublist
    r2_score_list.append(r2_score(y, y_pred))
    print(' - r2_score:')
    print(r2_score(y, y_pred))

plt.savefig('n_est_correlation.png')