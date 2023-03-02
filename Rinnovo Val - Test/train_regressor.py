import pandas as pd
import numpy as np
import joblib as jl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from predict_samples import predict_samples

def train_regressor(data_folder, metadata, estimators, window_size, method, reg_path):

    hp_tot_lists = []

    for i in range (1, metadata.shape[0]+1):
        _, hp_tot_list = predict_samples(data_folder, metadata, estimators, window_size, method, i)
        hp_tot_lists.append([hp_tot_list])

    lin_reg = LinearRegression()

    # Create two arrays to shuffle
    X = np.array(hp_tot_lists)
    y = np.array(metadata['aha'].values)

    # Generate a permutation index and use it to shuffle both arrays
    permutation_idx = np.random.permutation(len(X))
    X_shuffled = X[permutation_idx]
    y_shuffled = y[permutation_idx]

    lin_reg.fit(X_shuffled, y_shuffled)

    jl.dump(lin_reg, reg_path)