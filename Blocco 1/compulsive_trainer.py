from train_kmeans_model import train_kmeans_model

#folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'

#FIRST CRASH: manca k4_w600_dtw_dba, k4_w600_dtw_mean, k4_w600_euclidean_dba, k4_w600_euclidean_mean
#             manca k4_w900_dtw_dba, k4_w900_dtw_mean, k4_w900_euclidean_dba, k4_w900_euclidean_mean

l_window_size=[300,600,900] # 3 values
l_n_clusters=[5] # 1 values
l_init_algorithm=['kmeans++'] # 1 values
l_max_iter=[30]
l_metric=['euclidean'] # 1 values
l_averaging_method=['mean'] # 1 values

models_number = len(l_window_size) * len(l_n_clusters) * len(l_init_algorithm) * len(l_max_iter) * len(l_metric) * len(l_averaging_method)
i = 1


for size in l_window_size:
    for n in l_n_clusters:
        for alg in l_init_algorithm:
            for it in l_max_iter:
                for metric in l_metric:
                    for avg in l_averaging_method:
                        train_kmeans_model(folder, size, n, alg, it, metric, avg)
                        print("Trained " + str(i) + " models out of " + str(models_number) + " (" + str(i/models_number*100) + " %)")
                        i += 1
