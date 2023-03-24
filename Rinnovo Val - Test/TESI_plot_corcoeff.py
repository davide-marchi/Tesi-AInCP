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


print(predictions_dataframe)

print((np.corrcoef(predictions_dataframe['CPI'].values, predictions_dataframe['AHA'].values))[0][1])

plt.scatter(predictions_dataframe['AHA'].values, predictions_dataframe['CPI'].values, c = predictions_dataframe['MACS'].values)
plt.xlabel('AHA')
plt.ylabel('CPI')
#plt.legend(['a'], ['b'], ['c'])
plt.show()
plt.close()