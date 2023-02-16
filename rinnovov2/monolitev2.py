import json
import os
import hashlib
from sktime.base import BaseEstimator
from train_clustering_model import train_clustering_model
from test_clustering_model import test_clustering_model
from patients_dashboard import patients_dashboard

#data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

models_files = []
models_files.append("modelsv2.json")

for file in models_files:

    # Load JSON data from file
    with open(file) as f:
        data = f.read()
    
    # Convert JSON data into Model instances
    models_specs = json.loads(data)

    for specs in models_specs:

        model_folder = "Trained_models/" + specs["method"] + "/" + str(specs["window_size"]) + "_seconds/" + str(specs["patients"]) + "_patients/" + specs["class_type"].split(".")[-1] + "/" + "model_" + hashlib.sha256(json.dumps(specs["params"], sort_keys=True).encode()).hexdigest()[:10] + "/"
        
        if not(os.path.exists(model_folder + "trained_model.zip")):

            print("Model not found -> Training started\n")
            train_clustering_model(data_folder, specs, model_folder, False)

        model = BaseEstimator().load_from_path(model_folder + 'trained_model.zip')

        print("Testing on AHA sessions:\n")
        #test_clustering_model(data_folder, model_name,  model["method"] , model_folder)

        print("Testing on WEEK sessions:\n")
        #patients_dashboard(data_folder, model_name, model["method"], model_folder)