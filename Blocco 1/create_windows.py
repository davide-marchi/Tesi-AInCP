import pandas as pd
import math
import numpy as np

def create_windows(WINDOW_SIZE, folder, patients):
    series = []
    y_AHA = []
    y_MACS =[]
    lost = 0
    total = 0
    taken = 0
    metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')

    for j in range (1,patients+1):
        df = pd.read_csv(folder + 'data/' + str(j) + '_AHA_1sec.csv')
        total += df.shape[0]

        #print('Paziente ' + str(j) + ' -> df.shape[0] = ' + str(df.shape[0]))

        # Nel caso in cui non bastasse una duplicazione dell'intera time series questa verrà scartata
        if df.shape[0]<WINDOW_SIZE:
            df_concat = pd.concat([df, df.iloc[:WINDOW_SIZE-df.shape[0]]], ignore_index = True, axis = 0)
            df = df_concat
            #print('MODIFICATO Paziente ' + str(j) + ' -> df.shape[0] = ' + str(df.shape[0]))

        scart = (df.shape[0] % WINDOW_SIZE)/2
        
        df_cut = df.iloc[math.ceil(scart):df.shape[0]-math.floor(scart)]
        lost += df.shape[0]-df_cut.shape[0]
        # Calculating magnitude
        magnitude_D = np.sqrt(np.square(df_cut['x_D']) + np.square(df_cut['y_D']) + np.square(df_cut['z_D']))
        magnitude_ND = np.sqrt(np.square(df_cut['x_ND']) + np.square(df_cut['y_ND']) + np.square(df_cut['z_ND']))
        for i in range (0, len(magnitude_D), WINDOW_SIZE):
            chunk_D = magnitude_D.iloc[i:i + WINDOW_SIZE]
            chunk_ND = magnitude_ND.iloc[i:i + WINDOW_SIZE]
            # Concat
            magnitude_concat = pd.concat([chunk_D, chunk_ND], ignore_index = True)
            series.append(magnitude_concat)
            y_AHA.append(metadata['AHA'].iloc[j-1])
            y_MACS.append(metadata['MACS'].iloc[j-1])
            taken += len(magnitude_concat)/2
    
    return series, y_AHA, y_MACS, total, taken, lost
