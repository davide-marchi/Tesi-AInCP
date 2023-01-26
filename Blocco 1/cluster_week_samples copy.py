import pandas as pd
import re
import joblib as jl
import os
import numpy as np
import matplotlib.pyplot as plt
from elaborate_magnitude import elaborate_magnitude



############
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'

model_name = 'KMEANS_K2_W600_kmeans++_euclidean_mean'

operation_type = 'concat'

model_folder = 'Blocco 1/'+ operation_type +'_version/60_patients/KMeans/' + model_name + '/'
############


def save_plots(metadata):
    metadata.plot.scatter(x='healthy_percentage', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AHA.png')
    metadata.plot.scatter(x='healthy_percentage', y='AI_week', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AI_week.png')
    metadata.plot.scatter(x='healthy_percentage', y='AI_aha', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AI_aha.png')
    #metadata.plot.scatter(x='AI_week', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_AI_week_AHA.png')
    #metadata.plot.scatter(x='AI_aha', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_AI_aha_AHA.png')


# Define the samples size
match = re.search(r"_W(\d+)", model_name)
folder_name = 'Blocco 1/'+ operation_type +'_version/week_predictions/' + model_name + '/'

if os.path.exists(folder_name + '/predictions_dataframe.csv'):
    print("Model already tested")
    save_plots(pd.read_csv(folder_name + '/predictions_dataframe.csv'))
    exit(0)
elif match and os.path.exists(model_folder + "trained_model"):
    model = jl.load(model_folder + "trained_model")
    sample_size = int(match.group(1))
else:
    print("Model not found or invalid sample size")
    exit(1)


metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')
metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

stats = pd.read_csv(model_folder + 'statistiche.csv')
hemi_cluster = int(float(stats['AHA'][2]) > float(stats['AHA'][3]))

guessed_hemiplegic_patients = 0
guessed_healthy_patients = 0
uncertain_patients = 0
guessed = []
healthy_percentage = []

for i in range (1,61):

    cluster_hemiplegic_samples = 0 #malati
    cluster_healthy_samples = 0 #sani
    series = []
    to_discard = []

    df = pd.read_csv(folder + 'data/' + str(i) + '_week_1sec.csv')
    
    print("Inizio fase chunking")

    magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
    magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

    for j in range (0, len(magnitude_D), sample_size):

        chunk_D = magnitude_D.iloc[j:j + sample_size]
        chunk_ND = magnitude_ND.iloc[j:j + sample_size]

        ####### CHUNK DELLA DIMENSIONE SBAGLIATA??????
        if chunk_D.size != sample_size:
            print("YOOOOOOOOOOOOOOO")# LO STAMPERà MAI???????????????

        series.append(elaborate_magnitude(operation_type, chunk_D, chunk_ND))

        if chunk_D.agg('sum') == 0 and chunk_ND.agg('sum') == 0:
            to_discard.append(int(j/sample_size))


    print("Inizio fase predizione")
    Y = model.predict(np.array(series))

    for index in to_discard:
        Y[index] = -1


    print("Inizio fase incrementi e stampe")
    for k in range(len(Y)):
        # Presupponendo che i pazienti emiplegici siano nel cluster 0
        if Y[k] == hemi_cluster:
            cluster_hemiplegic_samples += 1
            Y[k] = -1
        elif Y[k] != -1:
            cluster_healthy_samples += 1  
            Y[k] = 1
        else:
            Y[k] = 0



    trend_block_size = 36
    ############# ANDAMENTO A BLOCCHI #############
    h_perc_list = []
    subList = [Y[n:n+trend_block_size] for n in range(0, len(Y), trend_block_size)]
    for l in subList:
        n_hemi = l.tolist().count(-1)
        n_healthy = l.tolist().count(1)
        if (n_hemi == 0 and n_healthy == 0):
            h_perc_list.append(-1)
        else:
            h_perc_list.append((n_healthy / (n_hemi + n_healthy)) * 100)
    #####################################

    '''
    ############# ANDAMENTO SMOOTH #############
    h_perc_list_smooth = []
    subList_smooth = [Y[n:n+trend_block_size] for n in range(0, len(Y)-trend_block_size+1)]
    for l in subList_smooth:

        print(len(l))

        n_hemi = l.tolist().count(-1)
        n_healthy = l.tolist().count(1)
        if (n_hemi == 0 and n_healthy == 0):
            h_perc_list_smooth.append(-1)
        else:
            h_perc_list_smooth.append((n_healthy / (n_hemi + n_healthy)) * 100)
    #####################################
    '''

    fig, axs = plt.subplots(3)
    fig.suptitle('week trend')
    axs[0].plot(magnitude_D)
    axs[0].plot(magnitude_ND)
    axs[1].scatter(list(range(len(Y))), Y, c=Y, cmap='brg')
    #axs[1].scatter(list(range(len(Y))), list([0]*len(Y)), c=Y, cmap='brg')
    axs[2].plot(h_perc_list)
    plt.show()
    plt.close()
    fig1, axs1 = plt.subplots(2,3)
    fig1.suptitle('daily trend')
    for i in range(0,2):
        for j in range(0,3):
            subplots_day_D = [magnitude_D[n:n+int(len(magnitude_D)/6)] for n in range(0, len(magnitude_D), int(len(magnitude_D)/6))]
            subplots_day_ND = [magnitude_ND[n:n+int(len(magnitude_D)/6)] for n in range(0, len(magnitude_ND),int(len(magnitude_D)/6))]
            for l in subplots_day_D:
                for z in subplots_day_ND:
                    axs1[i][j].plot(l)
                    axs1[i][j].plot(z)
    plt.show()
    plt.close()

    


    is_hemiplegic = (metadata['hemi'].iloc[i-1] == 2)

    guess =not((cluster_hemiplegic_samples > cluster_healthy_samples) ^ is_hemiplegic) if cluster_hemiplegic_samples!=cluster_healthy_samples else 'uncertain'
    print('Patient ', i, ' guessed: ', guess)
    guessed.append(guess)

    if (cluster_healthy_samples == 0 and cluster_hemiplegic_samples == 0):
        healthy_percentage.append(np.nan)
    else:
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


os.makedirs(folder_name, exist_ok=True)
with open(folder_name + 'predictions_stats.txt', "w") as f:
    f.write("Guessed patients: " + str(guessed_healthy_patients + guessed_hemiplegic_patients) + "/60 (" + str(((guessed_healthy_patients + guessed_hemiplegic_patients)/60)*100) + "%)\n")
    f.write("Guessed hemiplegic patients: " + str(guessed_hemiplegic_patients) + "/34 (" + str((guessed_hemiplegic_patients/34)*100) + "%)\n")
    f.write("Guessed healthy patients: " + str(guessed_healthy_patients) + "/26 (" + str((guessed_healthy_patients/26)*100) + "%)\n")
    f.write("Uncertain patients: " + str(uncertain_patients) + "/60 (" + str((uncertain_patients/60)*100) + "%)")

metadata.to_csv(folder_name + 'predictions_dataframe.csv')
save_plots(metadata)