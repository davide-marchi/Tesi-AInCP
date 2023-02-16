import json
import os
from train_clustering_model import train_clustering_model
from test_clustering_model import test_clustering_model
from patients_dashboard import patients_dashboard

#data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

models_files = []
models_files.append("models.json")

for file in models_files:

    # Load JSON data from file
    with open("models.json") as f:
        data = f.read()
    
    # Convert JSON data into Model instances
    models = json.loads(data)

    for model in models:

        model_folder = "Trained_models/" + model["method"] + "/" + model["window_size"] + "/" + model["patients"] + "_patients/" + model["class_type"].split(".")[-1] + "/"
        model_name = model["name"]

        if not(os.path.exists(model_folder + model_name)):

            print("Model not found -> Training started\n")
            train_clustering_model(data_folder, False, model, model_name)

        print("Testing on AHA sessions:\n")
        test_clustering_model(data_folder, model_name,  model["method"] , model_folder)

        print("Testing on WEEK sessions:\n")
        patients_dashboard(data_folder, model_name, model["method"], model_folder)