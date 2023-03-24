import hashlib
import os
import json
import pandas as pd
from sktime.base import BaseEstimator
from train_regressor import train_regressor
import joblib as jl
import numpy as np
from predict_samples import predict_samples
import datetime
import matplotlib
import matplotlib.pyplot as plt
from create_windows import create_windows
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.base import BaseClusterer
from sklearn.metrics import f1_score
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm
from sklearn.model_selection import train_test_split

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm
from sktime.datasets import load_arrow_head




if os.getlogin() == 'david':
    data_folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
elif os.getlogin() == 'giord':
    data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
else:
    data_folder = input("Enter the data folder path: ")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

X_cw, y_AHA, _, y = create_windows(data_folder, 'ai', 300)

estimator = TimeSeriesKMedoids().load_from_path('Trained_models/ai/300_seconds/TimeSeriesKMedoids/gridsearch_c42fbe82af/best_estimator.zip')

print(estimator.is_fitted)

print(type(X_cw))
print('-----------------------------------------------')

plot_cluster_algorithm(estimator, X_cw, estimator.n_clusters)

X, y = load_arrow_head(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print(type(X))
print(type(X_test))

k_medoids = TimeSeriesKMedoids(
    n_clusters=5,  # Number of desired centers
    init_algorithm="forgy",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    metric="dtw",  # Distance metric to use
    random_state=1,
)
k_medoids.fit(X_train)
plot_cluster_algorithm(k_medoids, X_test, k_medoids.n_clusters)

plot_cluster_algorithm(estimator, X_cw, estimator.n_clusters)


#y_pred = estimator.predict(X)
#inverted_y_pred = [1 if item == 0 else 0 for item in y_pred]

#hemi_cluster = 1
#if issubclass(type(estimator), BaseClusterer) and f1_score(y, y_pred, average='weighted') < f1_score(y, inverted_y_pred, average='weighted'):
#    hemi_cluster = 0