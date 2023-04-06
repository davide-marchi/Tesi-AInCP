import pandas as pd
import matplotlib.pyplot as plt
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'


df = pd.read_excel(folder+'metadata2022_04.xlsx')

print(df.describe())

ai_aha = df['AI_aha']
ai_week = df['AI_week']
macs = df['MACS']

#plt.scatter(x=ai_aha, y=ai_week, c=macs.astype('category').cat.codes)
plt.scatter(x=ai_aha, y=ai_week, c=macs)
plt.xlabel("AI AHA",
            fontweight ='bold', 
            size=14)
plt.ylabel("AI WEEK", 
            fontweight ='bold',
            size=14)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='-', color='k', lw=1, scalex=False, scaley=False)
plt.grid()
plt.show()