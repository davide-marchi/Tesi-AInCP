import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv('Rinnovo Val - Test/week_stats/predictions_dataframe.csv')

print(df['predicted_aha'].values)
print(type(df['predicted_aha'].values))
print(df['AHA'].values)
print(type(df['AHA'].values))
print(np.corrcoef(df['predicted_aha'], df['AHA']))
print(r2_score(df['AHA'].values, df['predicted_aha'].values))