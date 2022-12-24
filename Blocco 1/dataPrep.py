import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creating dataframe
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/'
df = pd.read_csv(folder + 'only AC/data/9_AHA_1sec.csv')

# Calculating magnitude
magnitude_D = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
magnitude_ND = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))

# Concat
mag_concat = pd.concat([magnitude_D, magnitude_ND], ignore_index = True)

# Polotting the concatted data
plt.plot(mag_concat)
plt.show()