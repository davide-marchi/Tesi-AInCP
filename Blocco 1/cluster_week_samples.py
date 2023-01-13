import pandas as pd
import re
import joblib as jl
import os


############
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
model_folder = 'Blocco 1/60_patients/KMeans/KMEANS_K2_W900_I30_kmeans++_euclidean_mean/'
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

guessed_patients = 0

for i in range (1,61):

    df = pd.read_csv(folder + 'data/' + str(i) + '_week_1sec.csv', chunksize=sample_size)
    week_lenght = df.shape[0]

    cluster_0_samples = 0
    cluster_1_samples = 0

    print(df)

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