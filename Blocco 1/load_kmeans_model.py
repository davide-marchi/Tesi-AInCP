import pandas as pd
import joblib as jl
import numpy as np
import matplotlib.pyplot as plt
from utils import create_windows
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm
from matplotlib.ticker import StrMethodFormatter


# Creating dataframe
#folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')


series, y, total, taken, lost = create_windows(900, folder)

print("taken = " + str(taken) + "  lost = " + str(lost) + "  on a total of = "+str(total)+"  percent lost = " + str((lost/total)*100) +"%")

# Create a DataFrame with a single column of Series objects
X = pd.DataFrame({'series': series})


k_means = jl.load('KMEANS_K4_W900_I10_kmeans++_dtw_dba')
y_pred = k_means.predict(X)
print(y_pred)
print(y)


stats = pd.DataFrame()
stats['cluster'] = y_pred
stats['AHA'] = y


# Group the DataFrame
grouped = stats.groupby(['cluster'])

# Compute the mean and median of the groups
mean_med_var_std = grouped.agg(['mean', 'median', 'var', 'std'])
print(mean_med_var_std)


grouped_stats = stats.groupby('cluster').agg(list)
print(grouped_stats)


print('score: ', k_means.score(X))


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


plot_cluster_algorithm(k_means, X, k_means.n_clusters)
