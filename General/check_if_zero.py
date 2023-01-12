import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/data/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/data/'

for j in range (1,61):
    df = pd.read_csv(folder + str(j) + '_AHA_1sec.csv')
    # Calculating magnitude
    magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
    magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))
    for i in range (0, len(magnitude_D), 300):
        chunk_D = magnitude_D[i:i + 300]
        chunk_ND = magnitude_ND[i:i + 300]
        # Concat
        magnitude_concat = pd.concat([chunk_D, chunk_ND], ignore_index = True)
        if len(magnitude_concat) == 600:
            sum = 0
            for x in magnitude_concat:
                sum += x
            print("paziente numero "+ str(j) + " :somma dei punti = " + str(sum))