import pandas as pd
import re
import joblib as jl
import os
import numpy as np


############
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
#folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
model_folder = 'Blocco 1/60_patients/KMeans/KMEANS_K2_W900_I10_kmeans++_dtw_dba/'
############

metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')

# Define the samples size
match = re.search(r"_W(\d+)", model_folder)
if match and os.path.exists(model_folder + "trained_model"):
    model = jl.load(model_folder + "trained_model")
    sample_size = int(match.group(1))
else:
    print("Model not found or invalid sample size")
    exit(1)

metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')

guessed_hemiplegic_patients = 0
guessed_healthy_patients = 0
uncertain_patients = 0
predictions = []

for i in range (1,61):

    df = pd.read_csv(folder + 'data/' + str(i) + '_week_1sec.csv', chunksize=sample_size)

    cluster_hemiplegic_samples = 0 #malati
    cluster_healthy_samples = 0 #sani
    
    for chunk in df:

        magnitude_D = np.sqrt(np.square(chunk['x_D']) + np.square(chunk['y_D']) + np.square(chunk['z_D']))
        magnitude_ND = np.sqrt(np.square(chunk['x_ND']) + np.square(chunk['y_ND']) + np.square(chunk['z_ND']))
        magnitude_concat = pd.concat([magnitude_D, magnitude_ND], ignore_index = True)

        if magnitude_concat.agg('sum') != 0:

            # Presupponendo che i pazienti emiplegici siano nel cluster 1
            if model.predict(np.array([magnitude_concat]))[0] == 1:
                cluster_hemiplegic_samples += 1
            else:
                cluster_healthy_samples += 1    

    is_hemiplegic = (metadata['hemi'].iloc[i-1] == 2)

    prediction = "Patient " + str(i) + " - Is hemiplegic? " + str(is_hemiplegic) + " - Predicted hemiplegic? " + str(cluster_hemiplegic_samples > cluster_healthy_samples)
    print(prediction)
    predictions.append(prediction)

    guessed_hemiplegic_patients += int(cluster_hemiplegic_samples > cluster_healthy_samples and is_hemiplegic)
    guessed_healthy_patients += int(cluster_hemiplegic_samples < cluster_healthy_samples and not is_hemiplegic)
    uncertain_patients += int(cluster_hemiplegic_samples == cluster_healthy_samples)


with open(model_folder + '/week_predictions.txt', 'w') as f:
    f.write("Guessed patients: " + str(guessed_healthy_patients + guessed_hemiplegic_patients) + "/60 (" + str(((guessed_healthy_patients + guessed_hemiplegic_patients)/60)*100) + "%)\n")
    f.write("Guessed hemiplegic patients: " + str(guessed_hemiplegic_patients) + "/34 (" + str((guessed_hemiplegic_patients/34)*100) + "%)\n")
    f.write("Guessed healthy patients: " + str(guessed_healthy_patients) + "/26 (" + str((guessed_healthy_patients/26)*100) + "%)\n")
    f.write("Uncertain patients: " + str(uncertain_patients) + "/60 (" + str((uncertain_patients/60)*100) + "%)\n\n")
    for prediction in predictions:
        f.write("%s\n" % prediction)