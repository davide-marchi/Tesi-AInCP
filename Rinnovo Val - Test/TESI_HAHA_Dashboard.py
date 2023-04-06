import hashlib
import os
import json
import pandas as pd
from sktime.base import BaseEstimator
from train_regressor import train_regressor
import joblib as jl
import numpy as np
from predict_samples import predict_samples
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings 

warnings.filterwarnings("ignore")


if os.getlogin() == 'david':
    data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
elif os.getlogin() == 'giord':
    data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
else:
    data_folder = input("Enter the data folder path: ")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

best_estimators_df = pd.read_csv('best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

#estimators_specs_list = []
#estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'concat'].iloc[0])
#estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'ai'].iloc[0])
#estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'difference'].iloc[0])
#estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= 0.975) & (best_estimators_df['window_size'] == 600)].iterrows()]
#estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] == 1) & (best_estimators_df['method'] == 'difference')].iterrows()]

#estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= 0.954) & (best_estimators_df['window_size'] == 300)].iterrows()]
estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= 0.939) & (best_estimators_df['window_size'] == 300)].iterrows()]


print('Expected estimators: ',len(estimators_specs_list))
estimators_list = []
model_id_concat = ''

for estimators_specs in estimators_specs_list:
    estimator_dir = "Trained_models/" + estimators_specs['method'] + "/" + str(estimators_specs['window_size']) + "_seconds/" + estimators_specs['model_type'].split(".")[-1] + "/gridsearch_" + estimators_specs['gridsearch_hash']  + "/"

    with open(estimator_dir + 'GridSearchCV_stats/best_estimator_stats.json', "r") as stats_f:
        grid_search_best_params = json.load(stats_f)

    estimator = BaseEstimator().load_from_path(estimator_dir + 'best_estimator.zip')
    estimators_list.append({'estimator': estimator, 'method': estimators_specs['method'], 'window_size': estimators_specs['window_size'], 'hemi_cluster': grid_search_best_params['Hemi cluster']})
    print('Loaded -> ', estimator_dir + 'best_estimator.zip')
    model_id_concat = model_id_concat + estimator_dir

print('Loaded estimators: ',len(estimators_list))

metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx')
metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

reg_path = 'Regressors/regressor_'+ (hashlib.sha256((model_id_concat).encode()).hexdigest()[:10])

if not(os.path.exists(reg_path)):
    #Dobbiamo allenare il regressore
    print('INIZIO TRAINING REGRESSORE')
    train_regressor(data_folder, metadata, estimators_list, reg_path)

regressor = jl.load(reg_path)
print('REGRESSOR READY')

stats_folder = 'Immagini_tesi/week_stats_graphs'
os.makedirs(stats_folder, exist_ok=True)
if not os.path.exists('timestamps_list'):
    start = datetime.datetime(2023, 1, 1)
    end = datetime.datetime(2023, 1, 7)
    step = datetime.timedelta(seconds=1)
    timestamps = []
    current = start
    while current < end:
        timestamps.append(matplotlib.dates.date2num(current))
        current += step
    jl.dump(timestamps, 'timestamps_list')
else:
    timestamps = jl.load('timestamps_list')

sample_size = estimators_list[0]['window_size']

trend_block_size = int((60 * 60 * 6) / sample_size)  # Numero di finestre (da 300/600/900 secondi) raggruppate in un blocco da 6 ore
significativity_threshold = 75                  # Percentuale di finestre in un blocco che devono essere prese per renderlo significativo

plot_show = False
'''
answer = input("Do you want to see the dashboard for each patient? (yes/no): \n")
# If the user enters "yes", show the plot
if answer.lower() == "yes":
    plot_show = True
'''

healthy_percentage = []
predicted_aha_list = []



for i in [25]:
    predictions, hp_tot_list, magnitude_D, magnitude_ND = predict_samples(data_folder, metadata, estimators_list, i)
    healthy_percentage.append(hp_tot_list)
    real_aha = metadata['AHA'].iloc[i-1]
    predicted_aha = regressor.predict(np.array([hp_tot_list]))[0]
    predicted_aha = 100 if predicted_aha > 100 else predicted_aha
    predicted_aha_list.append(predicted_aha)

    print('Patient ', i)
    print(' - AHA:     ', real_aha)
    print(' - HP:      ', hp_tot_list)
    print(' - AHA predicted from HP: ', predicted_aha)


    #################### ANDAMENTO WEEK GENERALE ####################

    #plt.title('Andamento magnitudo')
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    plt.plot(timestamps, magnitude_D)
    plt.plot(timestamps, magnitude_ND)
    plt.xlabel("Orario")
    plt.ylabel("Magnitudo")
    plt.gcf().set_size_inches(8, 2)
    plt.tight_layout()
    plt.savefig(stats_folder + '/5est_subject_' +str(i)+'_mag.png', dpi = 500)
    plt.close()

    # Fase di plotting
    #fig, axs = plt.subplots(7)
    #fig.suptitle('Patient ' + str(i) + ' week trend, AHA: ' + str(real_aha))
        #axs[0].xaxis.set_minor_locator(matplotlib.dates.HourLocator())
        #axs[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(12))
        #axs[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=6))
        #axs[0].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%H-%M'))
    #axs[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%H:%M'))
    #axs[0].plot(timestamps, magnitude_D)
    #axs[0].plot(timestamps, magnitude_ND)


    ########################## AI PLOT ##########################
    ai_list = []
    subList_magD = [magnitude_D[n:n+(trend_block_size*sample_size)] for n in range(0, len(magnitude_D), trend_block_size*sample_size)]
    subList_magND = [magnitude_ND[n:n+(trend_block_size*sample_size)] for n in range(0, len(magnitude_ND), trend_block_size*sample_size)]
    for l in range(len(subList_magD)):        
        if (subList_magD[l].mean() + subList_magND[l].mean()) == 0:
            ai_list.append(np.nan)
        else:
            ai_list.append(((subList_magD[l].mean() - subList_magND[l].mean()) / (subList_magD[l].mean() + subList_magND[l].mean())) * 100)

    #axs[1].grid()
    #axs[1].set_ylim([-101,101])
    #axs[1].plot(ai_list)

    #plt.title('Andamento AI')
    plt.xlabel("Orario")
    plt.ylabel("Asimmetry Index")
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    plt.plot(timestamps[::21600], ai_list)
    plt.gcf().set_size_inches(8, 2)
    plt.tight_layout()
    plt.savefig(stats_folder + '/5est_subject_' +str(i)+'_AI.png', dpi = 500)
    plt.close()


    #################### GRAFICO DEI PUNTI ####################
    for pred in predictions:
        #axs[2].scatter(list(range(len(pred))), pred, c=pred, cmap='brg', s=1) 
        plt.scatter(list(range(len(pred))), pred, c=pred, cmap='brg', s=1)

    #plt.title('Grafico delle predizioni')
    plt.xlabel("Sample")
    plt.ylabel("Classificazione")
    plt.gcf().set_size_inches(8, 2)
    plt.tight_layout()
    plt.savefig(stats_folder + '/5est_subject_' +str(i)+'_samples.png', dpi = 500)
    plt.close()

    #################### ANDAMENTO A BLOCCHI ####################

    for pred in predictions:
        h_perc_list = []
        subList = [pred[n:n+trend_block_size] for n in range(0, len(pred), trend_block_size)]
        for l in subList:
            n_hemi = l.tolist().count(-1)
            n_healthy = l.tolist().count(1)
            if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_threshold:
                h_perc_list.append(np.nan)
            else:
                h_perc_list.append((n_healthy / (n_hemi + n_healthy)) * 100)

        #h_perc_list.append(h_perc_list[-1]) PER LA LINEA ORIZZONTALE FINALE
        #axs[4].grid()
        #axs[4].set_ylim([-1,101])
        #axs[4].plot(h_perc_list, drawstyle = 'steps-post')
        plt.grid()
        ax = plt.gca()
        ax.set_ylim([-1,101])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.plot(timestamps[::21600], h_perc_list, drawstyle = 'steps-post')
        
    #plt.title('Andamento CPI su finestre disgiunte')
    plt.xlabel("Orario")
    plt.ylabel("CPI")
    plt.gcf().set_size_inches(8, 2)
    plt.tight_layout()
    plt.savefig(stats_folder + '/5est_subject_' +str(i)+'_CPIblocks.png', dpi = 500)
    plt.close()
    
    ##################### ANDAMENTO SMOOTH ######################
    h_perc_list_smooth_list = []
    #plt.title('Andamento CPI su finestra scorrevole')
    plt.grid()
    ax = plt.gca()
    ax.set_ylim([-1,101])
    for pred in predictions:
        h_perc_list_smooth = []
        h_perc_list_smooth_significativity = []
        subList_smooth = [pred[n:n+trend_block_size] for n in range(0, len(pred)-trend_block_size+1)]
        for l in subList_smooth:
            n_hemi = l.tolist().count(-1)
            n_healthy = l.tolist().count(1)
            h_perc_list_smooth_significativity.append(((n_hemi + n_healthy) / trend_block_size) * 100)
            if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_threshold:
                h_perc_list_smooth.append(np.nan)
            else:
                h_perc_list_smooth.append((n_healthy / (n_hemi + n_healthy)) * 100)

        #axs[5].plot(h_perc_list_smooth)

        h_perc_list_smooth_list.append(h_perc_list_smooth)

        plot_h_perc_list_smooth = [np.nan] * (6*12-1) + h_perc_list_smooth

        ax = plt.gca()
        ax.set_ylim([-1,101])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.plot(timestamps[::300], plot_h_perc_list_smooth)


    plt.xlabel("Orario")
    plt.ylabel("CPI")
    plt.gcf().set_size_inches(8, 2)
    plt.tight_layout()
    plt.savefig(stats_folder + '/5est_subject_' +str(i)+'_CPIsmooth.png', dpi = 500)
    plt.close()

    ##################### SIGNIFICATIVITY PLOT ####################

    #plt.title('Grafico della significatività')
    plt.grid()
    ax = plt.gca()
    ax.set_ylim([-1,101])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    plt.axhline(y = significativity_threshold, color = 'r', linestyle = '-', label='Soglia')
    plot_h_perc_list_smooth_significativity = [np.nan] * (6*12-1) + h_perc_list_smooth_significativity
    plt.plot(timestamps[::300], plot_h_perc_list_smooth_significativity)
    plt.legend()
    plt.xlabel("Orario")
    plt.ylabel("Significatività")
    plt.gcf().set_size_inches(8, 2)
    plt.tight_layout()
    plt.savefig(stats_folder + '/5est_subject_' +str(i)+'_sig.png', dpi = 500)
    plt.close()

    ##################### PREDICTED AHA PLOT ####################


    aha_list_smooth = []
    for elements in zip(*h_perc_list_smooth_list):
        if np.isnan(elements[0]):
            aha_list_smooth.append(np.nan)
        else:
            predicted_window_aha = (regressor.predict(np.array([elements])))
            aha_list_smooth.append(predicted_window_aha if predicted_window_aha <= 100 else 100)

    #plt.title('Andamento Home-AHA')
    conf = 5
    plt.grid()
    ax = plt.gca()
    ax.set_ylim([-1,101])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    plt.axhline(y = real_aha, color = 'b', linestyle = '--', linewidth= 1, label='AHA')
    plt.xlabel("Orario")
    plt.ylabel("Home-AHA")
    plt.plot(timestamps[::300], [np.nan] * (6*12-1) + aha_list_smooth, c = 'grey')
    plt.plot(timestamps[::300], [np.nan] * (6*12-1) + [x if real_aha + conf < x else np.nan for x in aha_list_smooth], c ='green')
    plt.plot(timestamps[::300], [np.nan] * (6*12-1) + [x if real_aha + 2*conf < x else np.nan for x in aha_list_smooth], c ='darkgreen')
    plt.plot(timestamps[::300], [np.nan] * (6*12-1) + [x if x < real_aha - conf else np.nan for x in aha_list_smooth], c ='orange')
    plt.plot(timestamps[::300], [np.nan] * (6*12-1) + [x if x < real_aha - 2*conf else np.nan for x in aha_list_smooth], c ='darkorange')
    plt.legend()
    plt.gcf().set_size_inches(8, 2)
    plt.tight_layout()
    plt.savefig(stats_folder + '/5est_subject_' +str(i)+'_Home-AHA.png', dpi = 500)
    plt.close()
    #############################################################
    
    #plt.savefig(stats_folder + '/subject_' +str(i)+'.png', dpi = 500)

    if(plot_show == True):
        plt.show() 
    plt.close()

#metadata['healthy_percentage'] = healthy_percentage
#metadata['predicted_aha'] = predicted_aha_list

#metadata.to_csv(stats_folder + '/predictions_dataframe.csv')

#metadata.plot.scatter(x='healthy_percentage', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(stats_folder + 'plot_healthyPerc_AHA.png')
#metadata.plot.scatter(x='healthy_percentage', y='AI_week', c='MACS', colormap='viridis').get_figure().savefig(stats_folder + 'plot_healthyPerc_AI_week.png')
#metadata.plot.scatter(x='healthy_percentage', y='AI_aha', c='MACS', colormap='viridis').get_figure().savefig(stats_folder + 'plot_healthyPerc_AI_aha.png')


#print("Coefficiente di Pearson tra hp e aha:          ", (np.corrcoef(metadata['healthy_percentage'], metadata['AHA'].values))[0][1])

