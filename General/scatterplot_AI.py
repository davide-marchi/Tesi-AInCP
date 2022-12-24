import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('only AC/metadata2022_04.xlsx')

print(df.describe())

ai_aha = df['AI_aha']
ai_week = df['AI_week']
macs = df['MACS']

#plt.scatter(x=ai_aha, y=ai_week, c=macs.astype('category').cat.codes)
plt.scatter(x=ai_aha, y=ai_week, c=macs)
plt.xlabel("AHA",
            fontweight ='bold', 
            size=14)
plt.ylabel("WEEK", 
            fontweight ='bold',
            size=14)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='-', color='k', lw=1, scalex=False, scaley=False)
plt.grid()
plt.show()