import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

def regression_and_correlation(path_to_csv,number_of_patients):
    # Reading the dataframe
    #df = pd.read_csv('Blocco 1/concat_version/week_predictions/KMEANS_K2_W600_kmeans++_dtw_dba/predictions_dataframe.csv', nrows=60)
    df = pd.read_csv(path_to_csv, nrows=number_of_patients)

    X = df[['healthy_percentage']].values
    y = df['AHA'].values

    # Coefficiente di correlazione = 0.89 per 60 pazienti e 0.37477669 per 34 pazienti
    print("\nCoefficiente di Pearson: ", (np.corrcoef(df['healthy_percentage'].values, df['AHA'].values))[0][1], '\n')


    # Create the linear regression object
    lin_reg = LinearRegression()

    # Create the k-fold cross validation object
    kf = KFold(n_splits=4, shuffle=True)

    # Compute the cross validation scores
    scores = cross_val_score(lin_reg, X, y, cv=kf)

    #print scores
    #print('Scores = ', scores)
    #print('Mean absolute error = ', np.mean(np.absolute(scores)))
    #print('RMSE = ', np.sqrt(np.mean(np.absolute(scores))))


    #############################################################################################

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    lin_reg.fit(X_train, y_train)

    #predictions = lin_reg.predict(X_test)
    #for i in range(len(predictions)):
        #print('AHA was ', y_test[i], ' predicted ', predictions[i])

    # Plot the data points
    plt.scatter(X, y)

    # Plot the regression line
    plt.plot(X, lin_reg.predict(X), color='red')

    # Add labels and show the plot
    plt.xlabel('healthy_percentage')
    plt.ylabel('AHA')
    plt.show()
    plt.close()

l_operation_type =      ['concat', 'difference','ai']   # ['concat','difference', 'ai']
l_patients =            [60]                             # [34, 60]
l_model_type =          ['KMeans']                       # ['KMeans','KMedoids']
l_window_size =         [300,600,900]                    # [300, 600, 900] 
l_metric =              ['euclidean_mean','dtw_dba']               # ['euclidean_mean', 'dtw_dba']


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