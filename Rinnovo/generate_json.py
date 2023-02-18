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



'''
preparazione per gridsearch:

    -togliere l'opzione 60-34 pazienti ? 

    -ha senso mettere tutti i parametri nel json o mettiamo solo quelli che ci interessano ?

    -ha senso togliere il json e fare due for nel main ?



    PARTE COMUNE usare generate_json.py che invece che params ha un dizionario di liste con tutte le permutazioni tipo:
    "params": {
            "average_params": [null],
            "averaging_method": ["mean", "dba"],
            "distance_params": [null],
            "init_algorithm": ["kmeans++", "forgy", "random"],
            "max_iter": [300],
            "metric": ["euclidean", "dtw"],
            "n_clusters": [2],
            "n_init": [10],
            "random_state": [null],
            "tol": [1e-06],
            "verbose": [false]
        }



    supervised:
        GridSearchCV(knn, param_grid, cv=KFold(n_splits=5)) dove param_grid è un dizionario di liste che contengono tutte le permutazioni per quella chiave
        avremmo quindi 9 modelli per ogni classificatore , ognuno il migliore del suo tipo
        
    unsupervised:
        fare una gridsearch simile a sopra però specificare uno scorer personalizzato che calcola la purity in base alla predizione hemi/nothemi (tipo percentuale di quelli azzeccati)
        avremmo quindi 9 modelli per ogni clusterer , ognuno il migliore del suo tipo


    esempio gridsearch supervised :
        from sklearn.model_selection import GridSearchCV
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        knn = KNeighborsTimeSeriesClassifier()
        param_grid = {"n_neighbors": [1, 5], "distance": ["euclidean", "dtw"]}
        parameter_tuning_method = GridSearchCV(knn, param_grid, cv=KFold(n_splits=4))

        parameter_tuning_method.fit(arrow_train_X, arrow_train_y)
        y_pred = parameter_tuning_method.predict(arrow_test_X)

        accuracy_score(arrow_test_y, y_pred)

    esempio gridsearch unsupervised :
        from sklearn.metrics import r2_score

        def scorer_f(estimator, X_train,Y_train):
            y_pred=estimator.predict(Xtrain)
            return r2_score(Y_train, y_pred)
        
        clf = IForest(random_state=47, behaviour='new',
              n_jobs=-1)

        param_grid = {'n_estimators': [20,40,70,100], 
                    'max_samples': [10,20,40,60], 
                    'contamination': [0.1, 0.01, 0.001], 
                    'max_features': [5,15,30], 
                    'bootstrap': [True, False]}

        grid_estimator = model_selection.GridSearchCV(clf, 
                                                    param_grid,
                                                    scoring=scorer_f,
                                                    cv=5,
                                                    n_jobs=-1,
                                                    return_train_score=True,
                                                    error_score='raise',
                                                    verbose=3)

        grid_estimator.fit(X_train, y_train)
'''