import json
import os
import hashlib
import itertools
import pandas as pd
from sktime.base import BaseEstimator
from train_best_model import train_best_model
from save_AHA_stats import save_AHA_stats
from save_week_stats import save_week_stats

if os.getlogin() == 'david':
    data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
elif os.getlogin() == 'giord':
    data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
else:
    data_folder = input("Enter the data folder path: ")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

kmeans_type = 'sktime.clustering.k_means.TimeSeriesKMeans'
kmeans_params =  {'averaging_method': ['mean'], 'init_algorithm': ['kmeans++', 'forgy'], 'metric': ['euclidean'], 'n_clusters': [2]}
kmeans = (kmeans_type, kmeans_params)

kmedoids_type = 'sktime.clustering.k_medoids.TimeSeriesKMedoids'
kmedoids_params = {'distance_params': [None], 'init_algorithm': ['random'], 'max_iter': [300], 'metric': ['dtw'], 'n_clusters': [8], 'n_init': [10], 'random_state': [None], 'tol': [1e-06], 'verbose': [False]}
kmedoids = (kmedoids_type, kmedoids_params)

cnn_type = 'sktime.classification.deep_learning.cnn.CNNClassifier'
cnn_params =  {'activation': ['sigmoid'], 'avg_pool_size': [3], 'batch_size': [16], 'callbacks': [None], 'kernel_size': [7], 'loss': ['mean_squared_error'], 'metrics': [None], 'n_conv_layers': [2], 'n_epochs': [2000], 'optimizer': [None], 'random_state': [None], 'use_bias': [True], 'verbose': [False]}
cnn = (cnn_type, cnn_params)

boss_type = 'sktime.classification.dictionary_based._boss.BOSSEnsemble'
boss_params = {'alphabet_size': [4], 'feature_selection': ['none'], 'max_ensemble_size': [500], 'max_win_len_prop': [1], 'min_window': [10], 'n_jobs': [1], 'random_state': [None], 'save_train_predictions': [False], 'threshold': [0.92], 'typed_dict': [True], 'use_boss_distance': [True]}
boss = (boss_type, boss_params)

shapedtw_type = 'sktime.classification.distance_based._shape_dtw.ShapeDTW'
shapedtw_params =  {'shape_descriptor_function': ['raw', 'paa'] }
shapedtw = (shapedtw_type, shapedtw_params)

l_method =              ['concat', 'ai']               # ['concat','difference', 'ai']
l_window_size =         [600, 900]                               # [300, 600, 900]
l_gridsearch_specs =    [kmeans]                                    # [kmeans, kmedoids, cnn, boss, shapedtw]

estimators_l = []
best_estimators_l = []

for method, window_size, gridsearch_specs in itertools.product(l_method, l_window_size, l_gridsearch_specs):

    print('Method: ', method, '\nWindow size: ', window_size, '\nGrid search specs: ', gridsearch_specs)

    model_type, model_params = gridsearch_specs

    gridsearch_folder = "Trained_models/" + method + "/" + str(window_size) + "_seconds/" + model_type.split(".")[-1] + "/" + "gridsearch_" + hashlib.sha256(json.dumps(model_params, sort_keys=True).encode()).hexdigest()[:10] + "/"

    if not(os.path.exists(gridsearch_folder + "best_estimator.zip")) or not(os.path.exists(gridsearch_folder + 'GridSearchCV_stats/cv_results.csv')):

        #print("Model not found -> Training started\n")
        train_best_model(data_folder, gridsearch_folder, model_type, model_params, method, window_size)

    cv_results = pd.read_csv(gridsearch_folder + 'GridSearchCV_stats/cv_results.csv', index_col=0)
    cv_results.columns = cv_results.columns.str.strip()
    cv_results['method'] = method
    cv_results['window_size'] = window_size
    cv_results['model_type'] = model_type.split(".")[-1]

    print('\n\n----------cv_results--------------\n\n')
    print(cv_results.dtypes)
    print('\n\n-----------estimators-----------\n\n')
    print(estimators_l)
    print('\n\n--------concat_estimators----------------\n\n')
    estimators_l.append(cv_results)
    best_estimators_l.append(cv_results.iloc[[cv_results['rank_test_score'].argmin()]])
    print(estimators_l)

estimators_df = pd.concat(estimators_l, ignore_index=True)
best_estimators_df = pd.concat(best_estimators_l, ignore_index=True)

print('\n\n------------------------\n\n')

estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv('estimators_results.csv')
best_estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv('best_estimators_results.csv')
