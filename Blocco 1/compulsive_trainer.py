from train_clustering_model import train_clustering_model

folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'

l_patients = [60,34]
l_model_type=['KMeans', 'KMedoids'] #'KMeans', 'KMedoids', 'KernelKMeans'
l_window_size=[300,600,900] # 3 values
l_n_clusters=[2,3,4,5] # 1 values
l_init_algorithm=['kmeans++'] # 1 values
l_metric=['euclidean', 'dtw'] # 1 values
l_averaging_method=['dba', 'mean'] # 1 values



models_number = len(l_patients)* len(l_model_type) * len(l_window_size) * len(l_n_clusters) * len(l_init_algorithm) * len(l_metric) * len(l_averaging_method)
i = 1

for patients in l_patients:
    for model_type in l_model_type:
        for size in l_window_size:
            for n in l_n_clusters:
                for alg in l_init_algorithm:
                    for metric in l_metric:
                        for avg in l_averaging_method:
                            if not((metric == 'euclidean' and avg == 'dba') or (metric == 'dtw' and avg == 'mean')):
                                train_clustering_model(model_type,folder,patients, size, n, alg, metric, avg)
                                print("Trained " + str(i) + " models out of " + str(models_number) + " (" + str(i/models_number*100) + " %)")    
                            i += 1
                                
