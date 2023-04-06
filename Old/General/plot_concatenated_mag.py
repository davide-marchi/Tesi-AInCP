import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib

# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/data/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/data/'

base_df = pd.read_csv(folder + str(21) + '_week_1sec.csv')


date = []

N = 1
first_cols = ['datetime']
df = base_df.groupby(base_df.index // N).agg({
    # Cols not to sum
    **{k: 'first' for k in first_cols},
    # Sum all other cols
    **{k: 'sum' for k in base_df.columns if k not in first_cols}
})



magnitude_D = np.sqrt(df['x_D']**2 + df['y_D']**2 + df['z_D']**2)
magnitude_ND = np.sqrt(df['x_ND']**2 + df['y_ND']**2 + df['z_ND']**2)
   
#mag_diff = magnitude_D - magnitude_ND

df['Magnitude_D'] = magnitude_D
df['Magnitude_ND'] = magnitude_ND



print(df.describe())

ai = ((magnitude_D.mean() - magnitude_ND.mean()) / (magnitude_D.mean() + magnitude_ND.mean())) * 100

print('AI_AHA: ')
print(ai)

#df.plot(x='datetime', y=['Magnitude_D', 'Magnitude_ND'], kind='line', grid= True, rot=90)

for str in df['datetime']:
    date.append(matplotlib.dates.date2num(datetime.strptime(str, '%Y-%m-%d %H:%M:%S')))

ax = plt.gca()
#ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
#ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%d'))
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%H'))
ax.tick_params(pad=10)

plt.plot(date, magnitude_D)
plt.plot(date, magnitude_ND)
#plt.gcf().autofmt_xdate()
plt.show()   


