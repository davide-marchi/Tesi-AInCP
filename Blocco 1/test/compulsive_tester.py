from test_clustering_model import test_clustering_model

############
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
############

plot_clusters = False

l_operation_type =      ['concat']   # ['concat','difference', 'ai']
l_patients =            [60]                             # [34, 60]
l_model_type =          ['KMeans']                       # ['KMeans','KMedoids']
l_window_size =         [300,600,900]                    # [300, 600, 900] 
l_metric =              ['euclidean_mean']               # ['euclidean_mean', 'dtw_dba']


models_number = len(l_operation_type) *len(l_patients)* len(l_model_type) * len(l_window_size) * len(l_metric)
i = 1

for operation_type in l_operation_type: 
    for patients in l_patients:
        for model_type in l_model_type:
            for window_size in l_window_size:
                for metric in l_metric:
                    model_name = model_type.upper()+'_K'+str(2)+'_W'+str(window_size)+'_kmeans++_'+metric
                    print(model_name)
                    model_folder = 'Blocco 1/'+ operation_type +'_version/'+str(patients)+'_patients/'+model_type+'/' + model_name + '/'
                    test_clustering_model(folder, model_name, operation_type, model_folder)
                    print("Tested " + str(i) + " models out of " + str(models_number) + " (" + str(i/models_number*100) + " %)")    
                    i += 1