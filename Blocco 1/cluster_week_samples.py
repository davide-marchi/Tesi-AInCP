import pandas as pd
import re
import joblib as jl
import os
import numpy as np


############
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
#folder = 'C:/Users/giord/Downloads/only AC data/only AC/'

model_name = 'KMEANS_K2_W900_kmeans++_euclidean_mean'

model_folder = 'Blocco 1/60_patients/KMeans/' + model_name + '/'
############

# Define the samples size
match = re.search(r"_W(\d+)", model_folder)
if match and os.path.exists(model_folder + "trained_model"):
    model = jl.load(model_folder + "trained_model")
    sample_size = int(match.group(1))
else:
    print("Model not found or invalid sample size")
    exit(1)

metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')
metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

guessed_hemiplegic_patients = 0
guessed_healthy_patients = 0
uncertain_patients = 0
guessed = []
healthy_percentage = []

for i in range (1,61):

    df = pd.read_csv(folder + 'data/' + str(i) + '_week_1sec.csv', chunksize=sample_size)

    cluster_hemiplegic_samples = 0 #malati
    cluster_healthy_samples = 0 #sani
    series = []
    
    print("Inizio fase chunking")
    for chunk in df:

        magnitude_D = np.sqrt(np.square(chunk['x_D']) + np.square(chunk['y_D']) + np.square(chunk['z_D']))
        magnitude_ND = np.sqrt(np.square(chunk['x_ND']) + np.square(chunk['y_ND']) + np.square(chunk['z_ND']))
        magnitude_concat = pd.concat([magnitude_D, magnitude_ND], ignore_index = True)

        if magnitude_concat.agg('sum') != 0:
            series.append(magnitude_concat)
    

    print("Inizio fase predizione")
    Y = model.predict(np.array(series))

    print("Inizio fase incrementi e stampe")
    for y in Y:
        # Presupponendo che i pazienti emiplegici siano nel cluster 0
        if y == 0:
            cluster_hemiplegic_samples += 1
        else:
            cluster_healthy_samples += 1    

    is_hemiplegic = (metadata['hemi'].iloc[i-1] == 2)

    guess = not((cluster_hemiplegic_samples > cluster_healthy_samples) ^ is_hemiplegic)
    print('Patient ', i, ' guessed: ', guess)
    guessed.append(guess)
    healthy_percentage.append((cluster_healthy_samples / (cluster_hemiplegic_samples + cluster_healthy_samples)) * 100)

    if cluster_hemiplegic_samples > cluster_healthy_samples and is_hemiplegic:
        guessed_hemiplegic_patients += 1
    elif cluster_hemiplegic_samples < cluster_healthy_samples and not is_hemiplegic:
        guessed_healthy_patients += 1
    elif cluster_hemiplegic_samples == cluster_healthy_samples:
        uncertain_patients += 1


metadata['guessed'] = guessed
metadata['healthy_percentage'] = healthy_percentage
print(metadata)

folder_name = 'Blocco 1/week_predictions/' + model_name + '/'
os.makedirs(folder_name, exist_ok=True)
with open(folder_name + 'predictions_stats.txt', "w") as f:
    f.write("Guessed patients: " + str(guessed_healthy_patients + guessed_hemiplegic_patients) + "/60 (" + str(((guessed_healthy_patients + guessed_hemiplegic_patients)/60)*100) + "%)\n")
    f.write("Guessed hemiplegic patients: " + str(guessed_hemiplegic_patients) + "/34 (" + str((guessed_hemiplegic_patients/34)*100) + "%)\n")
    f.write("Guessed healthy patients: " + str(guessed_healthy_patients) + "/26 (" + str((guessed_healthy_patients/26)*100) + "%)\n")
    f.write("Uncertain patients: " + str(uncertain_patients) + "/60 (" + str((uncertain_patients/60)*100) + "%)")

metadata.to_csv(folder_name + 'predictions_dataframe.csv')

metadata.plot.scatter(x='AHA', y='healthy_percentage', c='MACS').get_figure().savefig(folder_name + 'plot_AHA_healthyPerc.png')