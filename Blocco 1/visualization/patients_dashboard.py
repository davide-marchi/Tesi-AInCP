import pandas as pd
import re
import joblib as jl
import os
import numpy as np
import matplotlib.pyplot as plt
from elaborate_magnitude import elaborate_magnitude
import datetime
import matplotlib



############
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'

model_name = 'KMEANS_K2_W600_kmeans++_euclidean_mean'

operation_type = 'concat'

model_folder = 'Blocco 1/'+ operation_type +'_version/60_patients/KMeans/' + model_name + '/'
############
    


def plot_daily_trend():
    ################################################### DAILY TREND
    sub_Y = np.array_split(Y, 6)
    fig1, axs1 = plt.subplots(2,3)
    fig1.suptitle('daily trend')
    subplots_day_D = [magnitude_D[n:n+int(len(magnitude_D)/6)] for n in range(0, len(magnitude_D), int(len(magnitude_D)/6))]
    subplots_day_ND = [magnitude_ND[n:n+int(len(magnitude_D)/6)] for n in range(0, len(magnitude_ND),int(len(magnitude_D)/6))]
    kek=-3
    for z in range(0,2):
        kek+=3
        for a in range(0,3):
            sub_Y_multiplied=[]
            for el in sub_Y[a+kek]:
                for u in range(sample_size):
                    sub_Y_multiplied.append(el)
        
            #axs1[z, a].scatter(list(range(len(subplots_day_D[a+kek]))), subplots_day_D[a+kek], c=sub_Y_multiplied, cmap= 'brg', s=5, alpha=0.5)
            #axs1[z, a].scatter(list(range(len(subplots_day_ND[a+kek]))), subplots_day_D[a+kek], c=sub_Y_multiplied, cmap= 'brg',s=5, alpha=0.5)          
            
            subplotD = [subplots_day_D[a+kek].iloc[n:n+sample_size] for n in range(0, len(subplots_day_D[a+kek]), sample_size)]
            subplotND = [subplots_day_ND[a+kek].iloc[n:n+sample_size] for n in range(0, len(subplots_day_ND[a+kek]), sample_size)]
            for s in range(len(subplotD)):
                if sub_Y[a+kek][s] == 0:
                    axs1[z, a].plot(subplotD[s], color='darkred', alpha=0.5)
                    axs1[z, a].plot(subplotND[s], color='red', alpha=0.5)
                elif sub_Y[a+kek][s] == 1:
                    axs1[z, a].plot(subplotD[s], color='darkgreen', alpha=0.5)
                    axs1[z, a].plot(subplotND[s], color='green', alpha=0.5)
                else:
                    axs1[z, a].plot(subplotD[s], color='darkblue', alpha=0.5)
                    axs1[z, a].plot(subplotND[s], color='blue', alpha=0.5)

    plt.show()
    plt.close()
    ################################################### DAILY TREND  



# Define the samples size
match = re.search(r"_W(\d+)", model_name)

if not(match) or not(os.path.exists(model_folder + "trained_model")):
    print("Model not found or invalid sample size")
    exit(1)  

model = jl.load(model_folder + "trained_model")
sample_size = int(match.group(1))

metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')
metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

stats = pd.read_csv(model_folder + 'statistiche.csv')
hemi_cluster = int(float(stats['AHA'][2]) > float(stats['AHA'][3]))

if not os.path.exists('Blocco 1/visualization/regressor_' + model_name):
    print('linear regressor not found')
    exit(1)

lin_reg = jl.load('Blocco 1/visualization/regressor_' + model_name)


if not os.path.exists('Blocco 1/visualization/timestamps_list'):
    start = datetime.datetime(2023, 1, 1)
    end = datetime.datetime(2023, 1, 7)
    step = datetime.timedelta(seconds=1)
    timestamps = []
    current = start
    while current < end:
        timestamps.append(matplotlib.dates.date2num(current))
        current += step
    jl.dump(timestamps, 'Blocco 1/visualization/timestamps_list')
else:
    timestamps = jl.load('Blocco 1/visualization/timestamps_list')


guessed = []
healthy_percentage = []

for i in range (1,61):

    aha = metadata['AHA'].iloc[i-1]
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

        if chunk_D.size == sample_size and chunk_ND.size == sample_size:

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

    fig, axs = plt.subplots(7)
    fig.suptitle('Patient ' + str(i) + ' week trend, AHA: ' + str(aha))
    
    #axs[0].xaxis.set_minor_locator(matplotlib.dates.HourLocator())
    #axs[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(12))
    #axs[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=6))
    #axs[0].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%H-%M'))
    axs[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%H:%M'))
    axs[0].plot(timestamps, magnitude_D)
    axs[0].plot(timestamps, magnitude_ND)
    axs[1].scatter(list(range(len(Y))), Y, c=Y, cmap='brg', s=10)

    trend_block_size = 72           # Numero di finestre (da 600 secondi) raggruppate in un blocco
    significativity_treshold = 50   # Percentuale di finestre in un blocco che devono essere prese per renderlo significativo

    ############# ANDAMENTO A BLOCCHI #############
    h_perc_list = []
    subList = [Y[n:n+trend_block_size] for n in range(0, len(Y), trend_block_size)]
    for l in subList:
        n_hemi = l.tolist().count(-1)
        n_healthy = l.tolist().count(1)
        if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_treshold:
            h_perc_list.append(np.nan)
        else:
            h_perc_list.append((n_healthy / (n_hemi + n_healthy)) * 100)

    #axs[2].fill_between(list(range(len(h_perc_list))), 0, h_perc_list, alpha=0.5)
    axs[2].grid()
    #axs[2].set_xlim([0,11])      ##########
    axs[2].set_ylim([-1,101])
    axs[2].plot(h_perc_list)
    h_perc_list.append(h_perc_list[-1])
    axs[3].grid()
    axs[3].set_ylim([-1,101])
    axs[3].plot(h_perc_list, drawstyle = 'steps-post')
    #####################################################


    
    ###################### AI PLOT ###############################
    ai_list = []
    subList_magD = [magnitude_D[n:n+(trend_block_size*sample_size)] for n in range(0, len(magnitude_D), trend_block_size*sample_size)]
    subList_magND = [magnitude_ND[n:n+(trend_block_size*sample_size)] for n in range(0, len(magnitude_ND), trend_block_size*sample_size)]
    for l in range(len(subList_magD)):        
        if (subList_magD[l].mean() + subList_magND[l].mean()) == 0:
            ai_list.append(np.nan)
        else:
            ai_list.append(((subList_magD[l].mean() - subList_magND[l].mean()) / (subList_magD[l].mean() + subList_magND[l].mean())) * 100)

    axs[4].grid()
    axs[4].set_ylim([-101,101])
    axs[4].plot(ai_list)
    ####################################################
    
    
    ############# ANDAMENTO SMOOTH #############
    h_perc_list_smooth = []
    h_perc_list_smooth_significativity = []
    subList_smooth = [Y[n:n+trend_block_size] for n in range(0, len(Y)-trend_block_size+1)]
    for l in subList_smooth:
        n_hemi = l.tolist().count(-1)
        n_healthy = l.tolist().count(1)
        h_perc_list_smooth_significativity.append(((n_hemi + n_healthy) / trend_block_size) * 100)
        if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_treshold:
            h_perc_list_smooth.append(np.nan)
        else:
            h_perc_list_smooth.append((n_healthy / (n_hemi + n_healthy)) * 100)

    predicted_h_perc = lin_reg.predict([[aha]])
    print('hp predicted: ', predicted_h_perc, ' aha was: ', aha)
    conf = 10
    compare_hp_aha = [np.nan if predicted_h_perc - conf <= x <= predicted_h_perc + conf else x for x in h_perc_list_smooth]

    axs[5].grid()
    axs[5].set_ylim([-1,101])
    axs[5].plot(h_perc_list_smooth, c ='green')
    axs[5].plot(compare_hp_aha, c = 'red')

    axs[6].grid()
    axs[6].set_ylim([-1,101])
    axs[6].plot(h_perc_list_smooth_significativity)
    axs[6].axhline(y = significativity_treshold, color = 'r', linestyle = '-')
    #####################################
    
    plt.show()
    plt.close()

    #plot_daily_trend()
    
    

    is_hemiplegic = (metadata['hemi'].iloc[i-1] == 2)

    guess =not((cluster_hemiplegic_samples > cluster_healthy_samples) ^ is_hemiplegic) if cluster_hemiplegic_samples!=cluster_healthy_samples else 'uncertain'
    print('Patient ', i, ' guessed: ', guess)
    guessed.append(guess)

    if (cluster_healthy_samples == 0 and cluster_hemiplegic_samples == 0):
        healthy_percentage.append(np.nan)
    else:
        healthy_percentage.append((cluster_healthy_samples / (cluster_hemiplegic_samples + cluster_healthy_samples)) * 100)


metadata['guessed'] = guessed
metadata['healthy_percentage'] = healthy_percentage
print(metadata)

guessed_hemiplegic_patients = len(metadata[(metadata['hemi']==2) & (metadata['guessed']==True)])
guessed_healthy_patients = len(metadata[(metadata['hemi']==1) & (metadata['guessed']==True)])
uncertain_patients = guessed.count('uncertain')

folder_name = 'Blocco 1/'+ operation_type +'_version/week_predictions/' + model_name + '/'
os.makedirs(folder_name, exist_ok=True)
with open(folder_name + 'predictions_stats.txt', "w") as f:
    f.write("Guessed patients: " + str(guessed_healthy_patients + guessed_hemiplegic_patients) + "/60 (" + str(((guessed_healthy_patients + guessed_hemiplegic_patients)/60)*100) + "%)\n")
    f.write("Guessed hemiplegic patients: " + str(guessed_hemiplegic_patients) + "/34 (" + str((guessed_hemiplegic_patients/34)*100) + "%)\n")
    f.write("Guessed healthy patients: " + str(guessed_healthy_patients) + "/26 (" + str((guessed_healthy_patients/26)*100) + "%)\n")
    f.write("Uncertain patients: " + str(uncertain_patients) + "/60 (" + str((uncertain_patients/60)*100) + "%)")

metadata.to_csv(folder_name + 'predictions_dataframe.csv')

metadata.plot.scatter(x='healthy_percentage', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AHA.png')
metadata.plot.scatter(x='healthy_percentage', y='AI_week', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AI_week.png')
metadata.plot.scatter(x='healthy_percentage', y='AI_aha', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AI_aha.png')
#metadata.plot.scatter(x='AI_week', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_AI_week_AHA.png')
#metadata.plot.scatter(x='AI_aha', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_AI_aha_AHA.png')