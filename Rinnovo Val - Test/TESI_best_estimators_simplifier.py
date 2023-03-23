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

best_estimators_df = pd.read_csv('best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

best_estimators_df['mean_test_score'] = best_estimators_df['mean_test_score'].round(3)
best_estimators_df['std_test_score'] = best_estimators_df['std_test_score'].round(3)

modified_csv = best_estimators_df[['method', 'model_type', 'params',  'mean_test_score', 'std_test_score']]

print(modified_csv)
print(type(modified_csv))

modified_csv.to_csv('Immagini_tesi/gridsearch_results/simplified_best_estimators')