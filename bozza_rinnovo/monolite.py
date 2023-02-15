import json
import os
from train_clustering_model import train_clustering_model
from test_clustering_model import test_clustering_model
from patients_dashboard import patients_dashboard


folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'

models_data = [
    ("KMeans", 600, "unsupervised", 60, "concat", 'sktime.clustering.k_means.TimeSeriesKMeans', {'average_params': None, 'averaging_method': 'mean', 'distance_params': None, 'init_algorithm': 'kmeans++', 'max_iter': 300, 'metric': 'euclidean', 'n_clusters': 2, 'n_init': 10, 'random_state': None, 'tol': 1e-06, 'verbose': False} )
]


models = []
for name,window_size, type, patients, method,class_type, params in models_data:
    model = {
        "name": name,
        "window_size": window_size,
        "type": type,
        "patients": patients,
        "method": method,
        "class_type": class_type,
        "params": params
    }
    models.append(model)

models_json = json.dumps(models, indent=4)

with open("bozza_rinnovo/models.json", "w") as f:
    f.write(models_json)




# Load JSON data from file
with open("bozza_rinnovo/models.json") as f:
    data = f.read()

# Convert JSON data into Model instances
models = json.loads(data)

# Print the attributes of each Model instance
for model in models:
    '''
    if model["name"] is not None:
        print("name:", model["name"])
    if model["window_size"] is not None:
        print("window_size:", model["window_size"])
    if model["type"] is not None:
        print("type:", model["type"])
    if model["patients"] is not None:
        print("patients:", model["patients"])
    if model["method"] is not None:
        print("method:", model["method"])
    if model["params"] is not None:
        for key, value in model["params"].items():
            print("\t",key,": ", value)
    '''

    #trovare un modo per separare il nome del modello dai suoi parametri: 
    #per unificare questo processo per tutti i modelli non possiamo utilizzare model["params"]
    #altrimenti dovremmo metterli tutti nel nome, per renderlo unico

    model_name = model["name"].upper()+'_K'+str(2)+'_W'+str(model["window_size"])+'_'+model["params"]['init_algorithm']+'_'+model["params"]['metric']+'_'+model["params"]['averaging_method']
    print(model_name)
    model_folder = 'Blocco 1/'+ model["method"] +'_version/'+str(model["patients"])+'_patients/'+model["name"]+'/' + model_name + '/'
    if not(os.path.exists(model_folder + "trained_model")):
        print("modello non allenato, avvio del training:\n")
        train_clustering_model(folder, False, model, model_name)


    print("testing phase:\n")
    test_clustering_model(folder, model_name,  model["method"] , model_folder)


    patients_dashboard(folder, model_name, model["method"], model_folder)

    print("bye !")