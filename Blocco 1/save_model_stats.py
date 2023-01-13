import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from matplotlib.ticker import StrMethodFormatter

def save_model_stats(X, y_AHA, y_MACS, y_pred, model, modelname, NCLUSTERS,model_folder):

    stats = pd.DataFrame()
    stats['cluster'] = y_pred
    stats['AHA'] = y_AHA
    stats['MACS'] = y_MACS

    # Group the DataFrame
    grouped = stats.groupby(['cluster'])

    # Compute the mean and median of the groups
    mean_med_var_std = grouped.agg(['mean', 'median', 'var', 'std', 'count'])
    with open(model_folder + '/statistiche.csv', 'w') as f:
        mean_med_var_std.to_csv(model_folder + '/statistiche.csv')

    grouped_stats = stats.groupby('cluster').agg(list)
    with open(model_folder + '/classificazione.csv', 'w') as f:
        grouped_stats.to_csv(model_folder + '/classificazione.csv')

    with open(model_folder + '/parametri.txt', 'a') as f:
        f.write('inertia: ' + str(model.inertia_))
        #f.write('silhouette score: ' + str(silhouette_score(X, model.labels_)))

    ax = stats.hist(column='AHA', by='cluster', bins=np.linspace(0,100,51), grid=False, figsize=(8,10), layout=(NCLUSTERS,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)

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
    
    plt.savefig(model_folder + "/istogramma_AHA.png")
    plt.close()

    ax = stats.hist(column='MACS', by='cluster', bins=np.linspace(-0.5,3.5,5), grid=False, figsize=(8,10), layout=(NCLUSTERS,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
  
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
        x.set_xlabel("Manual Ability Classification System (MACS)", labelpad=20, weight='bold', size=12)

        # Set y-axis label
        if i == 1:
            x.set_ylabel("Clusters", labelpad=50, weight='bold', size=12)

        # Format y-axis label
        x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

        x.tick_params(axis='x', rotation=0)
    
    plt.savefig(model_folder + "/istogramma_MACS.png")
    plt.close()
