import numpy as np
import joblib as jl
from sklearn.linear_model import LinearRegression
from predict_samples import predict_samples
import os

def train_regressor(data_folder, metadata, estimators, reg_path):

    hp_tot_lists = []

    for i in range (1, metadata.shape[0]+1):
        _,hp_tot_list,_,_ = predict_samples(data_folder, metadata, estimators, i)
        hp_tot_lists.append(hp_tot_list)

    lin_reg = LinearRegression()

    #TODO: togliere quadre a riga 13
    #X = np.array(hp_tot_lists).reshape(metadata.shape[0], len(estimators))
    X = np.array(hp_tot_lists)
    y = np.array(metadata['AHA'].values)

    print(X)
    print(y)

    # Generate a permutation index and use it to shuffle both arrays
    permutation_idx = np.random.permutation(len(X))
    X_shuffled = X[permutation_idx]
    y_shuffled = y[permutation_idx]

    lin_reg.fit(X_shuffled, y_shuffled)

    os.makedirs('Regressors', exist_ok = True)
    jl.dump(lin_reg, reg_path)