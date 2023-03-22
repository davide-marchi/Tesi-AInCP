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

# Cambio la directory di esecuzione in quella dove si trovano i file utili
os.chdir("./Rinnovo Val - Test")

metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx')

df = pd.read_csv(data_folder + 'data/1_AHA_1sec.csv')

# Calculating magnitude
magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

date_time = df['datetime']
datetime_object = [datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S') for datetime_str in df['datetime']]

#print(date_time)
#print(type(date_time[0]))

plt.plot(magnitude_D)
plt.plot(magnitude_ND)
plt.show()
plt.close()

plt.plot(magnitude_D)
plt.plot(magnitude_ND)
plt.plot(elaborate_magnitude('difference', magnitude_D, magnitude_ND), label='Differenza')
plt.title("Andamento differenze")
plt.xlabel("Secondi di sessione AHA")
plt.ylabel("Magnitudo")

plt.show()
plt.close()

plt.plot(magnitude_D)
plt.plot(magnitude_ND)
plt.plot(elaborate_magnitude('concat', magnitude_D, magnitude_ND), label='Concatenazione')
plt.show()
plt.close()

plt.plot(magnitude_D)
plt.plot(magnitude_ND)
plt.plot(elaborate_magnitude('ai', magnitude_D, magnitude_ND), label='Asimmetry Index')
plt.show()
plt.close()