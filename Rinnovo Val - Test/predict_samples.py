import numpy as np
import pandas as pd
from elaborate_magnitude import elaborate_magnitude

def predict_samples(data_folder, metadata, estimators, patient):

    df = pd.read_csv(data_folder + 'data/' + str(patient) + '_week_1sec.csv')
    magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
    magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

    to_discard = []

    if len(set(es['window_size'] for es in estimators)) != 1:
        print('You have selected estimators that operates on different window sizes')
        exit(1)

    window_size = estimators[0]['window_size']

    for es in estimators:
        es['series'] = []

    # Fase di chunking
    for j in range (0, len(magnitude_D), window_size):

        chunk_D = magnitude_D.iloc[j:j + window_size]
        chunk_ND = magnitude_ND.iloc[j:j + window_size]

        if chunk_D.size == window_size and chunk_ND.size == window_size:
            
            for es in estimators:
                es['series'].append(elaborate_magnitude(es['method'], chunk_D, chunk_ND))

            if chunk_D.agg('sum') == 0 and chunk_ND.agg('sum') == 0:
                to_discard.append(int(j/window_size))

    is_hemiplegic = (metadata['hemi'].iloc[patient-1] == 2)

    y_list = []
    hp_tot_list = []
    
    # Fase di predizione
    for es in estimators:
        #print(np.array(es['series']).shape)
        y = es['estimator'].predict(np.array(es['series']))
        #print(es['method'])

        for index in to_discard:
            y[index] = -1

        hemi_cluster = es['hemi_cluster']

        cluster_healthy_samples = 0     # Non emiplegici
        cluster_hemiplegic_samples = 0  # Emiplegici

        for k in range(len(y)):
            if y[k] == hemi_cluster:
                cluster_hemiplegic_samples += 1
                y[k] = -1
            elif y[k] != -1:
                cluster_healthy_samples += 1  
                y[k] = 1
            else:
                y[k] = 0
        
        y_list.append(y)

        hp_tot = np.nan
        if (cluster_healthy_samples != 0 or cluster_hemiplegic_samples != 0):
            hp_tot = (cluster_healthy_samples / (cluster_hemiplegic_samples + cluster_healthy_samples)) * 100

        hp_tot_list.append(hp_tot)

    return y_list, hp_tot_list, magnitude_D, magnitude_ND