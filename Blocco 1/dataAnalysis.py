import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib

# Creating dataframe
#folder = 'C:/Users/giord/Downloads/only AC data/only AC/data/'
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/data/'


df = pd.read_csv(folder + str(21) + '_AHA_1sec.csv')


df['Magnitude_D'] = np.sqrt(np.square(df['x_D']) + np.square(df['y_D']) + np.square(df['z_D']))
df['Magnitude_ND'] = np.sqrt(np.square(df['x_ND']) + np.square(df['y_ND']) + np.square(df['z_ND']))
magnitude_concat = pd.concat([df['Magnitude_D'], df['Magnitude_ND']], ignore_index = True)


print(df)
print(df.describe())



plt.plot(magnitude_concat)
plt.show()   


