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

subject = 24

df = pd.read_csv(data_folder + 'data/'+ str(subject) + '_AHA_1sec.csv')

# Calculating magnitude
magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

date_time = df['datetime']
datetime_object = [datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S') for datetime_str in df['datetime']]

#print(date_time)
#print(type(date_time[0]))


plt.plot(magnitude_D, label = 'Mano dominante')
plt.plot(magnitude_ND, label = 'Mano non dominante')
plt.xlabel("Secondi")
plt.ylabel("Magnitudo")
plt.legend(loc='best')
plt.gcf().set_size_inches(8, 2)
plt.tight_layout()
plt.savefig('Immagini_tesi/Preprocessing/AHA_' + str(subject) +'_mag_D_mag_ND.png', dpi = 500)
plt.close()

plt.plot(elaborate_magnitude('concat', magnitude_D, magnitude_ND))
plt.xlabel("Secondi")
plt.ylabel("Magnitudo")
plt.gcf().set_size_inches(8, 2)
plt.tight_layout()
plt.savefig('Immagini_tesi/Preprocessing/AHA_' + str(subject) +'_Concatenazione.png', dpi = 500)
plt.close()

plt.plot(elaborate_magnitude('difference', magnitude_D, magnitude_ND))
plt.xlabel("Secondi")
plt.ylabel("Differenza magnitudo")
plt.gcf().set_size_inches(8, 2)
plt.tight_layout()
plt.savefig('Immagini_tesi/Preprocessing/AHA_' + str(subject) +'_Difference.png', dpi = 500)
plt.close()

plt.plot(elaborate_magnitude('ai', magnitude_D, magnitude_ND))
plt.xlabel("Secondi")
plt.ylabel("Asimmetry Index")
plt.gcf().set_size_inches(8, 2)
plt.tight_layout()
plt.savefig('Immagini_tesi/Preprocessing/AHA_' + str(subject) +'_AsimmetryIndex.png', dpi = 500)
plt.close()