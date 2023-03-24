import hashlib
import os
import json
import pandas as pd
from sktime.base import BaseEstimator
import joblib as jl
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

from train_regressor import train_regressor
from elaborate_magnitude import elaborate_magnitude
from predict_samples import predict_samples


if os.getlogin() == 'david':
    data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
elif os.getlogin() == 'giord':
    data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
else:
    data_folder = input("Enter the data folder path: ")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

predictions_dataframe = pd.read_csv('week_stats/predictions_dataframe.csv', index_col=0)

predictions_dataframe['CPI'] = [float(string.strip('[]')) for string in predictions_dataframe['healthy_percentage']]
print(type(predictions_dataframe['healthy_percentage'][0]))
print(type(predictions_dataframe['AHA'][0]))
print(type(predictions_dataframe['CPI'][0]))


predictions_dataframe['CPI'] = predictions_dataframe['CPI'].round(3)

#predictions_dataframe['std_test_score'] = best_estimators_df['std_test_score'].round(3)

print(predictions_dataframe[['subject', 'MACS', 'AHA', 'CPI']])

print('corrcoef CPI-AHA = ', (np.corrcoef(predictions_dataframe['CPI'].values, predictions_dataframe['AHA'].values))[0][1])
print('corrcoef AI_week-AHA = ', (np.corrcoef(predictions_dataframe['AI_week'].values, predictions_dataframe['AHA'].values))[0][1])






#plt.scatter(predictions_dataframe['AHA'].values, predictions_dataframe['CPI'].values, c = predictions_dataframe['MACS'].values)

scatter_x = np.array(predictions_dataframe['CPI'].values)
scatter_y = np.array(predictions_dataframe['AHA'].values)
group = np.array(predictions_dataframe['MACS'].values)
cdict = {0:'green', 1: 'gold', 2: 'orange', 3: 'red'}

#print(scatter_x)
#print(scatter_y)
#print(group)

fig, ax = plt.subplots()
ax.grid()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = 'MACS ' + str(g), s = 50)
ax.legend()

plt.xlabel('CPI')
plt.ylabel('AHA')
plt.savefig('Immagini_tesi/CPI/scatter_AHA_CPI_best300.png', dpi = 500)

#plt.xlabel('AHA')
#plt.ylabel('CPI')
#plt.legend(['a'], ['b'], ['c'])
#plt.show()
#plt.close()