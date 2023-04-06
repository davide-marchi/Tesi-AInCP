from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold


if os.getlogin() == 'david':
    data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
elif os.getlogin() == 'giord':
    data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
else:
    data_folder = input("Enter the data folder path: ")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx')
metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)


composed_dataframe = pd.DataFrame()
# Sperimentale per concatenare i CSV contenenti series
for i in range(1, metadata.shape[0] + 1):
    composed_dataframe = pd.concat([composed_dataframe, pd.read_csv('week_stats_300_5est/predictions_dataframe_' + str(i) +'.csv', index_col = False)])

#print(composed_dataframe)

#composed_dataframe['CPI_list'] = [json.loads(string) for string in composed_dataframe['healthy_percentage']]
CPI_list = [json.loads(string) for string in composed_dataframe['healthy_percentage']]
composed_dataframe.drop(['healthy_percentage'], axis=1, inplace=True)

#print(composed_dataframe)

print(type(CPI_list))
print(type(CPI_list[0]))
print(type(CPI_list[0][0]))

X = np.array(CPI_list)
y = composed_dataframe['AHA'].values

#print(X)
#print(type(X))
#print(X.shape)
#print(X[0])
#print(type(X[0]))
#print('---')
#print(y)
#print(type(y))


#print(composed_dataframe['CPI_list'].values[0])
#print(type(composed_dataframe['CPI_list'].values[0]))
#print(type(composed_dataframe['CPI_list'].values[0][0]))

model = LinearRegression()

#cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#kf = KFold(n_splits=5, shuffle=True)

rkf = RepeatedKFold(n_splits=5, n_repeats=1000)

score = cross_val_score(model, X, y, cv=rkf)

print(score)
print(np.average(score))
print(score.mean())

model.fit(X, y)

plt.scatter(composed_dataframe['AHA'].values, model.predict(X))

plt.grid()
plt.xlim([0, 120])
plt.ylim([0, 120])

plt.show()
plt.close()