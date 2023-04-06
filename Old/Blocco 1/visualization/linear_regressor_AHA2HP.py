import pandas as pd
import numpy as np
import joblib as jl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

def regression_and_correlation(path_to_csv,number_of_patients):
    # Reading the dataframe
    #df = pd.read_csv('Blocco 1/concat_version/week_predictions/KMEANS_K2_W600_kmeans++_dtw_dba/predictions_dataframe.csv', nrows=60)
    df = pd.read_csv(path_to_csv, nrows=number_of_patients)

    X = df[['AHA']].values
    y = df['healthy_percentage'].values


    # Create the linear regression object
    lin_reg = LinearRegression()



    #############################################################################################

    X_train, _, y_train, _ = train_test_split(X, y, test_size=1, shuffle=True)

    lin_reg.fit(X_train, y_train)

    jl.dump(lin_reg, 'Blocco 1/visualization/regressor_AHA2HP_KMEANS_K2_W600_kmeans++_euclidean_mean')

    #predictions = lin_reg.predict(X_test)
    #for i in range(len(predictions)):
        #print('AHA was ', y_test[i], ' predicted ', predictions[i])

    # Plot the data points
    plt.scatter(X, y)

    # Plot the regression line
    plt.plot(X, lin_reg.predict(X), color='red')

    # Add labels and show the plot
    plt.xlabel('AHA')
    plt.ylabel('healthy_percentage')
    plt.show()

l_operation_type =      ['concat']   # ['concat','difference', 'ai']
l_patients =            [60]                             # [34, 60]
l_model_type =          ['KMeans']                       # ['KMeans','KMedoids']
l_window_size =         [600]                    # [300, 600, 900] 
l_metric =              ['euclidean_mean']               # ['euclidean_mean', 'dtw_dba']


models_number = len(l_operation_type) *len(l_patients)* len(l_model_type) * len(l_window_size) * len(l_metric)
i = 1

for operation_type in l_operation_type: 
    for patients in l_patients:
        for model_type in l_model_type:
            for window_size in l_window_size:
                for metric in l_metric:
                    model_name = model_type.upper()+'_K'+str(2)+'_W'+str(window_size)+'_kmeans++_'+metric
                    print(model_name+' : '+operation_type+' : '+str(patients)+' patients')
                    model_folder = 'Blocco 1/'+ operation_type +'_version/week_predictions/' + model_name + '/predictions_dataframe.csv'
                    regression_and_correlation(model_folder, patients)
                    print(str(i) + " models out of " + str(models_number) + "\n")    
                    i += 1