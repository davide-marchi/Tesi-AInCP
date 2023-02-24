import json
import os
import hashlib
import itertools
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
kmeans_params =  {'averaging_method': ['mean'], 'init_algorithm': ['kmeans++'], 'metric': ['euclidean'], 'n_clusters': [2]}
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

l_method =              ['concat']               # ['concat','difference', 'ai']
l_window_size =         [600]                               # [300, 600, 900]
l_model_specs =         [kmeans]                                    # [kmeans, kmedoids, cnn, boss, shapedtw]

for method, window_size, model_specs in itertools.product(l_method, l_window_size, l_model_specs):

    print('method ', method, '\t\twindow_size ', window_size, '\t\tmodel_specs ', model_specs)

    model_type, model_params = model_specs

    model_folder = "Trained_models/" + method + "/" + str(window_size) + "_seconds/" + model_type.split(".")[-1] + "/" + "model_" + hashlib.sha256(json.dumps(model_params, sort_keys=True).encode()).hexdigest()[:10] + "/"

    if not(os.path.exists(model_folder + "model.zip")):

        #print("Model not found -> Training started\n")
        train_best_model(data_folder, model_folder, model_type, model_params, method, window_size)

    model = BaseEstimator().load_from_path(model_folder + 'model.zip')

    if not(os.path.exists(model_folder + "AHA_stats")):
        print("Inizio a fare statistiche su sessioni AHA")
        save_AHA_stats(model, model_folder + 'AHA_stats/', data_folder, method, window_size)
        #save_AHA_stats(X, y_AHA, y_MACS, y_pred, model, stats_folder)
    
    if not(os.path.exists(model_folder + "week_stats")):
        print("Inizio a fare statistiche su dati week")
        #print("Visualizing patients dashboard:\n")
        #save_week_stats(data_folder, model_name, model["method"], model_folder)
        save_week_stats(model, model_folder, data_folder, method, window_size)


    '''
    1. va bene la funzione di scoring?
    2. parlare della frammentazione
    3. test su AHA con cross validation
    4. KFold oppure stratifiedKFold?
    '''