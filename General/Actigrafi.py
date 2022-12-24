import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#headers = ['datetime', 'x_D']

df = pd.read_csv(r'only AC/data/9_AHA_1sec.csv')
print(df)


#x = df['datetime']
#y = np.linalg.norm([df['x_D'], df['y_D'], df['z_D']])
#plt.plot(x, y)

#plt.gcf().autofmt_xdate()




magnitude_D = np.sqrt(df['x_D']**2 + df['y_D']**2 + df['z_D']**2)
magnitude_ND = np.sqrt(df['x_ND']**2 + df['y_ND']**2 + df['z_ND']**2)
Ai = ((magnitude_D - magnitude_ND) / (magnitude_D + magnitude_ND)) * 100
mag_diff = magnitude_D - magnitude_ND

df['Magnitude_D'] = magnitude_D
df['Magnitude_ND'] = magnitude_ND
df['AI'] = Ai
df['Mag_diff'] = mag_diff
#print(df)

N = 60
first_cols = ['datetime']
res_df = df.groupby(df.index // N).agg({
    # Cols not to sum
    **{k: 'first' for k in first_cols},
    # Sum all other cols
    **{k: 'sum' for k in df.columns if k not in first_cols}
})
print(res_df) #la AI non ha valori sensati, è la somma delle AI

res_df.plot(x='datetime', y=['Magnitude_D', 'Magnitude_ND', 'Mag_diff', 'AI'], kind='line')
plt.show()

