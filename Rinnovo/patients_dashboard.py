# Importazione di librerie e funzioni
import re
import os
import datetime
import matplotlib
import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from elaborate_magnitude import elaborate_magnitude


# Funzione per prendere la media del primo quartile dei massimi valori di una lista
def first_quartile_of_max(my_list):
    sorted_list = sorted(my_list, reverse=True)
    quartile_length = len(sorted_list) // 2
    if quartile_length == 0:
        return sorted_list[0]
    first_quartile = sum(sorted_list[:quartile_length]) / quartile_length
    return first_quartile

def patients_dashboard(folder, model_name, operation_type, model_folder):

    plot_show = False
    answer = input("Do you want to see the dashboard for each patient? (yes/no): \n")
    # If the user enters "yes", show the plot
    if answer.lower() == "yes":
        plot_show = True

    # Lettura della dimensione della finestra, ricerca del modello e caricamento
    match = re.search(r"_W(\d+)", model_name)

    if not(match) or not(os.path.exists(model_folder + "trained_model")):
        print("Model not found or invalid sample size")
        exit(1)

    sample_size = int(match.group(1))
    model = jl.load(model_folder + "trained_model")
    if (not os.path.exists('Blocco 1/visualization/regressor_AHA2HP_' + model_name)) or (not os.path.exists('Blocco 1/visualization/regressor_HP2AHA_' + model_name)) :
        print('A linear regressor was not found')
        exit(1)

    lin_reg_AHA2HP = jl.load('Blocco 1/visualization/regressor_AHA2HP_' + model_name)
    lin_reg_HP2AHA = jl.load('Blocco 1/visualization/regressor_HP2AHA_' + model_name)

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

    metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')
    metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

    stats = pd.read_csv(model_folder + 'statistiche.csv')
    hemi_cluster = int(float(stats['AHA'][2]) > float(stats['AHA'][3]))

    guessed = []
    healthy_percentage = []
    max_hp_list = []
    max_quart_hp_list = []

    trend_block_size = 36            # Numero di finestre (da 600 secondi) raggruppate in un blocco
    significativity_threshold = 50   # Percentuale di finestre in un blocco che devono essere prese per renderlo significativo

    for i in range (1,61):

        df = pd.read_csv(folder + 'data/' + str(i) + '_week_1sec.csv')
        magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
        magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

        series = []
        to_discard = []

        # Fase di chunking
        for j in range (0, len(magnitude_D), sample_size):

            chunk_D = magnitude_D.iloc[j:j + sample_size]
            chunk_ND = magnitude_ND.iloc[j:j + sample_size]

            if chunk_D.size == sample_size and chunk_ND.size == sample_size:

                series.append(elaborate_magnitude(operation_type, chunk_D, chunk_ND))

                if chunk_D.agg('sum') == 0 and chunk_ND.agg('sum') == 0:
                    to_discard.append(int(j/sample_size))

        # Fase di predizione
        Y = model.predict(np.array(series))

        for index in to_discard:
            Y[index] = -1

        cluster_healthy_samples = 0     # Non emiplegici
        cluster_hemiplegic_samples = 0  # Emiplegici

        for k in range(len(Y)):
            if Y[k] == hemi_cluster:
                cluster_hemiplegic_samples += 1
                Y[k] = -1
            elif Y[k] != -1:
                cluster_healthy_samples += 1  
                Y[k] = 1
            else:
                Y[k] = 0

        aha = metadata['AHA'].iloc[i-1]
        is_hemiplegic = (metadata['hemi'].iloc[i-1] == 2)
        hp_tot = np.nan
        aha_from_hp_tot = np.nan
        guess = not((cluster_hemiplegic_samples > cluster_healthy_samples) ^ is_hemiplegic) if cluster_hemiplegic_samples!=cluster_healthy_samples else 'uncertain'
        guessed.append(guess)
        
        if (cluster_healthy_samples != 0 or cluster_hemiplegic_samples != 0):
            hp_tot = (cluster_healthy_samples / (cluster_hemiplegic_samples + cluster_healthy_samples)) * 100
            aha_from_hp_tot = (lin_reg_HP2AHA.predict([[hp_tot]]))[0] 
            aha_from_hp_tot = 100 if aha_from_hp_tot > 100 else aha_from_hp_tot

        healthy_percentage.append(hp_tot)
        predicted_h_perc = lin_reg_AHA2HP.predict([[aha]])

        print('Patient ', i)
        print(' - Guessed: ', guess)
        print(' - AHA:     ', aha)
        print(' - HP:      ', round(hp_tot, 2))
        print(' - AHA predicted from HP: ', round(aha_from_hp_tot, 2))
        print(' - HP predicted from AHA: ', round(predicted_h_perc[0], 2), '\n')

        # Fase di plotting
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

        #################### ANDAMENTO A BLOCCHI ####################
        h_perc_list = []
        subList = [Y[n:n+trend_block_size] for n in range(0, len(Y), trend_block_size)]
        for l in subList:
            n_hemi = l.tolist().count(-1)
            n_healthy = l.tolist().count(1)
            if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_threshold:
                h_perc_list.append(np.nan)
            else:
                h_perc_list.append((n_healthy / (n_hemi + n_healthy)) * 100)

        h_perc_list.append(h_perc_list[-1])
        axs[2].grid()
        axs[2].set_ylim([-1,101])
        axs[2].plot(h_perc_list, drawstyle = 'steps-post')
        #############################################################
        
        ########################## AI PLOT ##########################
        ai_list = []
        subList_magD = [magnitude_D[n:n+(trend_block_size*sample_size)] for n in range(0, len(magnitude_D), trend_block_size*sample_size)]
        subList_magND = [magnitude_ND[n:n+(trend_block_size*sample_size)] for n in range(0, len(magnitude_ND), trend_block_size*sample_size)]
        for l in range(len(subList_magD)):        
            if (subList_magD[l].mean() + subList_magND[l].mean()) == 0:
                ai_list.append(np.nan)
            else:
                ai_list.append(((subList_magD[l].mean() - subList_magND[l].mean()) / (subList_magD[l].mean() + subList_magND[l].mean())) * 100)

        axs[3].grid()
        axs[3].set_ylim([-101,101])
        axs[3].plot(ai_list)
        #############################################################
        
        ##################### ANDAMENTO SMOOTH ######################
        aha_list_smooth = []
        h_perc_list_smooth = []
        h_perc_list_smooth_significativity = []
        subList_smooth = [Y[n:n+trend_block_size] for n in range(0, len(Y)-trend_block_size+1)]
        for l in subList_smooth:
            n_hemi = l.tolist().count(-1)
            n_healthy = l.tolist().count(1)
            h_perc_list_smooth_significativity.append(((n_hemi + n_healthy) / trend_block_size) * 100)
            if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_threshold:
                h_perc_list_smooth.append(np.nan)
                aha_list_smooth.append(np.nan)
            else:
                h_perc_list_smooth.append((n_healthy / (n_hemi + n_healthy)) * 100)
                predicted_aha = (lin_reg_HP2AHA.predict([[h_perc_list_smooth[-1]]]))[0]
                aha_list_smooth.append(predicted_aha if predicted_aha <= 100 else 100) 
            #print("h_perc: ",h_perc_list_smooth[-1]," aha_pred: ",aha_list_smooth[-1], " diff: ",aha_list_smooth[-1]-h_perc_list_smooth[-1])

        conf = 5

        filtered_list = list(filter(lambda x: not np.isnan(x), h_perc_list_smooth))

        max_hp_list.append(max(filtered_list))
        max_quart_hp_list.append(first_quartile_of_max(filtered_list))

        axs[4].grid()
        axs[4].set_ylim([-1,101])
        axs[4].axhline(y = predicted_h_perc, color = 'b', linestyle = '--', linewidth= 1, label='predicted hp')
        axs[4].plot(h_perc_list_smooth, c ='green')
        axs[4].plot([np.nan if predicted_h_perc - conf <= x <= predicted_h_perc + conf else x for x in h_perc_list_smooth], c = 'red')
        axs[4].legend()

        axs[5].grid()
        axs[5].set_ylim([-1,101])
        axs[5].axhline(y = significativity_threshold, color = 'r', linestyle = '-', label='threshold')
        axs[5].plot(h_perc_list_smooth_significativity)
        axs[5].legend()
        #############################################################

        ##################### PREDICTED AHA PLOT ####################
        #new_list = [[i] for i in h_perc_list_smooth]
        axs[6].grid()
        axs[6].set_ylim([-1,101])
        axs[6].axhline(y = aha, color = 'b', linestyle = '--', linewidth= 1, label='aha')
        axs[6].plot(aha_list_smooth, c = 'green')
        axs[6].plot([np.nan if aha - conf <= x <= aha + conf else x for x in aha_list_smooth], c ='red')
        axs[6].legend()
        #############################################################
        
        if(plot_show==True):
            plt.show()
        plt.close()

    metadata['guessed'] = guessed
    metadata['healthy_percentage'] = healthy_percentage
    #print(metadata)

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

    print('Modello usato per calcolare hp: ', model_name)
    print("Coefficiente di Pearson tra hp e aha:          ", (np.corrcoef(metadata['healthy_percentage'], metadata['AHA'].values))[0][1])
    print("Coefficiente di Pearson tra max hp e aha:      ", (np.corrcoef(max_hp_list, metadata['AHA'].values))[0][1])
    print("Coefficiente di Pearson tra maxquart hp e aha: ", (np.corrcoef(max_quart_hp_list, metadata['AHA'].values))[0][1])