
#MAX ITERATION is fixed to 10

l_window_size=[300,600,900] # 3 values
l_n_clusters=[4,6,8] # 3 values
l_init_algorithm=['kmeans++'] # 1 values
l_metric=['euclidean', 'dtw'] # 2 values
l_averaging_method=['mean', 'dba'] # 2 values

#36 models 

for metric in l_metric:
    for n in l_n_clusters:
        for alg in l_init_algorithm:
            for avg in l_averaging_method:
                for size in l_window_size:
                    trainer(n, alg, metric, avg, size)