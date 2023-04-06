import pandas as pd
import re
import joblib as jl
import os
import numpy as np
from elaborate_magnitude import elaborate_magnitude

def save_plots(metadata, folder_name):
    metadata.plot.scatter(x='healthy_percentage', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AHA.png')
    metadata.plot.scatter(x='healthy_percentage', y='AI_week', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AI_week.png')
    metadata.plot.scatter(x='healthy_percentage', y='AI_aha', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_healthyPerc_AI_aha.png')
    #metadata.plot.scatter(x='AI_week', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_AI_week_AHA.png')
    #metadata.plot.scatter(x='AI_aha', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(folder_name + 'plot_AI_aha_AHA.png')

def test_clustering_model(folder, model_name, operation_type, model_folder):

    # Define the samples size
    match = re.search(r"_W(\d+)", model_name)
    folder_name = 'Blocco 1/'+ operation_type +'_version/week_predictions/' + model_name + '/'

    if os.path.exists(folder_name + '/predictions_dataframe.csv'):
        save_plots(pd.read_csv(folder_name + '/predictions_dataframe.csv'), folder_name)
        return
    elif match and os.path.exists(model_folder + "trained_model"):
        model = jl.load(model_folder + "trained_model")
        sample_size = int(match.group(1))
    else:
        print("Model not found or invalid sample size")
        return


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

        df = pd.read_csv(folder + 'data/' + str(i) + '_week_1sec.csv')

        cluster_hemiplegic_samples = 0 #malati
        cluster_healthy_samples = 0 #sani
        series = []

        magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
        magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

        for j in range (0, len(magnitude_D), sample_size):

            chunk_D = magnitude_D.iloc[j:j + sample_size]
            chunk_ND = magnitude_ND.iloc[j:j + sample_size]

            if chunk_D.agg('sum') != 0 or chunk_ND.agg('sum') != 0:
                series.append(elaborate_magnitude(operation_type, chunk_D, chunk_ND))

        Y = model.predict(np.array(series))

        for y in Y:
            # Presupponendo che i pazienti emiplegici siano nel cluster 0
            if y == hemi_cluster:
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


    os.makedirs(folder_name, exist_ok=True)
    with open(folder_name + 'predictions_stats.txt', "w") as f:
        f.write("Guessed patients: " + str(guessed_healthy_patients + guessed_hemiplegic_patients) + "/60 (" + str(((guessed_healthy_patients + guessed_hemiplegic_patients)/60)*100) + "%)\n")
        f.write("Guessed hemiplegic patients: " + str(guessed_hemiplegic_patients) + "/34 (" + str((guessed_hemiplegic_patients/34)*100) + "%)\n")
        f.write("Guessed healthy patients: " + str(guessed_healthy_patients) + "/26 (" + str((guessed_healthy_patients/26)*100) + "%)\n")
        f.write("Uncertain patients: " + str(uncertain_patients) + "/60 (" + str((uncertain_patients/60)*100) + "%)")

    metadata.to_csv(folder_name + 'predictions_dataframe.csv')
    save_plots(metadata, folder_name)