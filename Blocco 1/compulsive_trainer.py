from train_clustering_model import train_clustering_model

folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'

patients = 60

l_model_type=['KMeans'] #'KMeans', 'KMedoids', 'KernelKMeans'
l_window_size=[300,600,900] # 3 values
l_n_clusters=[2] # 1 values
l_init_algorithm=['kmeans++'] # 1 values
l_metric=['dtw'] # 1 values
l_averaging_method=['dba'] # 1 values



models_number = len(l_model_type) * len(l_window_size) * len(l_n_clusters) * len(l_init_algorithm) * len(l_metric) * len(l_averaging_method)
i = 1


for model_type in l_model_type:
    for size in l_window_size:
        for n in l_n_clusters:
            for alg in l_init_algorithm:
                for metric in l_metric:
                    for avg in l_averaging_method:
                        train_clustering_model(model_type,folder,patients, size, n, alg, metric, avg)
                        print("Trained " + str(i) + " models out of " + str(models_number) + " (" + str(i/models_number*100) + " %)")
                        i += 1
