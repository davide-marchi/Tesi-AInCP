from train_clustering_model import train_clustering_model

folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'

plot_clusters = False

l_operation_type =      ['difference']   # ['concat','difference', 'ai']
l_patients =            [34]                            # [34, 60]
l_model_type =          ['KMeans']                      # ['KMeans', 'KShapes', 'KernelKMeans', 'KMedoids']
l_window_size =         [900]                           # [300, 600, 900] 
l_n_clusters =          [29]                       # [2, 3, 4, 6, 8, 10, 12, 14, 16, 24, 32, 40, 60]
l_init_algorithm =      ['kmeans++']                    # ['kmeans++', 'forgy', 'random']
l_metric =              ['euclidean']            # ['euclidean', 'dtw']


models_number = len(l_operation_type) *len(l_patients)* len(l_model_type) * len(l_window_size) * len(l_n_clusters) * len(l_init_algorithm) * len(l_metric)
i = 1

for operation_type in l_operation_type: 
    for patients in l_patients:
        for model_type in l_model_type:
            for window_size in l_window_size:
                for n_clusters in l_n_clusters:
                    for init_algorithm in l_init_algorithm:
                        for metric in l_metric:
                            train_clustering_model(folder, plot_clusters, operation_type, patients, model_type, window_size, n_clusters, init_algorithm, metric)
                            print("Trained " + str(i) + " models out of " + str(models_number) + " (" + str(i/models_number*100) + " %)")    
                            i += 1