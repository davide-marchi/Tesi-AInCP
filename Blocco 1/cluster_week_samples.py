import pandas as pd
import re
import joblib as jl
import os
import numpy as np


############
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
model_folder = 'Blocco 1/60_patients/KMeans/KMEANS_K2_W900_I30_kmeans++_euclidean_mean/'
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

guessed_patients = 0
indecisi = 0

for i in range (1,61):

    df = pd.read_csv(folder + 'data/' + str(i) + '_week_1sec.csv', chunksize=sample_size)
    #week_lenght = df.shape[0]  

    cluster_0_samples = 0 #malati
    cluster_1_samples = 0 #sani
    series=[]
    
    for chunk in df:
        magnitude_D = np.sqrt(np.square(chunk['x_D']) + np.square(chunk['y_D']) + np.square(chunk['z_D']))
        magnitude_ND = np.sqrt(np.square(chunk['x_ND']) + np.square(chunk['y_ND']) + np.square(chunk['z_ND']))
        magnitude_concat = pd.concat([magnitude_D, magnitude_ND], ignore_index = True)
        series.append(magnitude_concat)
        
    X = pd.DataFrame({'series': series}) 
    prediction = model.predict(X)
    for n in prediction:
        if n == 0:
            cluster_0_samples+=1
        else:
            cluster_1_samples+=1    
    
    if cluster_0_samples > cluster_1_samples and metadata['hemi'].iloc[i-1] == 2:
        print("ho indovinato un paziente malato. bene !")
        guessed_patients+=1
    elif cluster_0_samples > cluster_1_samples and metadata['hemi'].iloc[i-1] == 1:
        print("ho sbagliato un paziente sano. pensavo fosse malato !")
    elif cluster_0_samples < cluster_1_samples and metadata['hemi'].iloc[i-1] == 1:
        print("ho indovinato un paziente sano. bene !")
        guessed_patients+=1
    elif cluster_0_samples < cluster_1_samples and metadata['hemi'].iloc[i-1] == 2:
        print("ho sbagliato un paziente malato. pensavo fosse sano !")
    elif cluster_0_samples == cluster_1_samples:
        print("eh qui non so decidermi molto bene.")
        indecisi+=1
        
print("guessed patients: " , guessed_patients, "/60 (", (guessed_patients/60)*100, "%)")
print("indecisi: " , indecisi, "/60 (", (indecisi/60)*100, "%)")

    # Iterate through the DataFrame
    #for j, rows in enumerate(df.iterrows()):
        #if j % sample_size == 0:
            #sample = df.iloc[j:j+sample_size]
            #print(type(sample))
            #print(sample)

            #calcolare magnitude D e ND

            #concatenare magnitude D e ND

            #Predirre la concatenazione
            #print(model.predict(pd.Series(sample)))

            #aumentare l'apposito contatore
    
    # Se sample sani > sample malati e il paziente è sano allora guessed_patients +=1

# stampare guessed_patients