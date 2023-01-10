import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def create_windows(WINDOW_SIZE, folder):
    series = []
    y = []
    lost = 0
    total = 0
    taken = 0
    metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')

    for j in range (1,61):
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
            y.append(metadata['AHA'].iloc[j-1])
            taken += len(magnitude_concat)/2
    
    return series, y, total, taken, lost


def save_model_stats(X, y, y_pred, k_means, modelname):

    stats = pd.DataFrame()
    stats['cluster'] = y_pred
    stats['AHA'] = y

    # Group the DataFrame
    grouped = stats.groupby(['cluster'])

    # Compute the mean and median of the groups
    mean_med_var_std = grouped.agg(['mean', 'median', 'var', 'std'])
    with open('./Blocco 1/' + modelname + '/statistiche.csv', 'w') as f:
        mean_med_var_std.to_csv('./Blocco 1/' + modelname + '/statistiche.csv')

    grouped_stats = stats.groupby('cluster').agg(list)
    with open('./Blocco 1/' + modelname + '/classificazione.csv', 'w') as f:
        grouped_stats.to_csv('./Blocco 1/' + modelname + '/classificazione.csv')

    with open('./Blocco 1/' + modelname + '/parametri.txt', 'a') as f:
        f.write('score: ' + str(k_means.score(X)))

    ax = stats.hist(column='AHA', by='cluster', bins=np.linspace(0,100,51), grid=False, figsize=(8,10), layout=(4,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)

    for i,x in enumerate(ax):

        # Despine
        x.spines['right'].set_visible(False)
        x.spines['top'].set_visible(False)
        x.spines['left'].set_visible(False)

        # Switch off ticks
        x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

        # Draw horizontal axis lines
        vals = x.get_yticks()
        for tick in vals:
            x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

        # Set x-axis label
        x.set_xlabel("Assisting Hand Assessment (AHA)", labelpad=20, weight='bold', size=12)

        # Set y-axis label
        if i == 1:
            x.set_ylabel("Clusters", labelpad=50, weight='bold', size=12)

        # Format y-axis label
        x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

        x.tick_params(axis='x', rotation=0)
    
    plt.savefig("./Blocco 1/" + modelname + "/istogramma.png")