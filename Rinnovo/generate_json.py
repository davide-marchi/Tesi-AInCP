import json
import os

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

kmeans_params =  {'average_params': None, 'averaging_method': 'mean', 'distance_params': None, 'init_algorithm': 'kmeans++', 'max_iter': 300, 'metric': 'euclidean', 'n_clusters': 2, 'n_init': 10, 'random_state': None, 'tol': 1e-06, 'verbose': False}
kmeans_type = 'sktime.clustering.k_means.TimeSeriesKMeans'
kmeans = (kmeans_type, kmeans_params)

kmedoids_params = {'distance_params': None, 'init_algorithm': 'random', 'max_iter': 300, 'metric': 'dtw', 'n_clusters': 8, 'n_init': 10, 'random_state': None, 'tol': 1e-06, 'verbose': False}
kmedoids_type = 'sktime.clustering.k_medoids.TimeSeriesKMedoids'
kmedoids = (kmedoids_type, kmedoids_params)

cnn_params =  {'activation': 'sigmoid', 'avg_pool_size': 3, 'batch_size': 16, 'callbacks': None, 'kernel_size': 7, 'loss': 'mean_squared_error', 'metrics': None, 'n_conv_layers': 2, 'n_epochs': 2000, 'optimizer': None, 'random_state': None, 'use_bias': True, 'verbose': False}
cnn_type = 'sktime.classification.deep_learning.cnn.CNNClassifier'
cnn = (cnn_type, cnn_params)

boss_params = {'alphabet_size': 4, 'feature_selection': 'none', 'max_ensemble_size': 500, 'max_win_len_prop': 1, 'min_window': 10, 'n_jobs': 1, 'random_state': None, 'save_train_predictions': False, 'threshold': 0.92, 'typed_dict': True, 'use_boss_distance': True}
boss_type = 'sktime.classification.dictionary_based._boss.BOSSEnsemble'
boss = (boss_type, boss_params)

shapedtw_params =  {'metric_params': None, 'n_neighbors': 1, 'shape_descriptor_function': 'raw', 'shape_descriptor_functions': None, 'subsequence_length': 30}
shapedtw_type = 'sktime.classification.distance_based._shape_dtw.ShapeDTW'
shapedtw = (shapedtw_type, shapedtw_params)

l_method =              ['concat','difference', 'ai']               # ['concat','difference', 'ai']
l_patients =            [60]                                        # [34, 60]
l_window_size =         [300,600,900]                               # [300, 600, 900] 
l_models =              [kmeans, kmedoids, cnn, boss, shapedtw]

models=[]
for method in l_method: 
    for patients in l_patients:
        for window_size in l_window_size:
            for type, params in l_models:
                
                model =    {
                    "window_size": window_size,
                    "patients": patients,
                    "method": method,
                    "class_type": type,
                    "params": params
                }
                
                models.append(model)


models_json = json.dumps(models, indent=4)

with open("input_models.json", "w") as f:
    f.write(models_json)
