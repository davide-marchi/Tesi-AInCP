import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Reading the dataframe
df = pd.read_csv('Blocco 1/concat_version/week_predictions/KMEANS_K2_W600_kmeans++_euclidean_mean/predictions_dataframe.csv', nrows=34)

X = df[['healthy_percentage']].values
y = df['AHA'].values


print(np.corrcoef(X[0], y)) #TODO

'''
# Create the linear regression object
lin_reg = LinearRegression()

# Create the k-fold cross validation object
kf = KFold(n_splits=4, shuffle=True)

# Compute the cross validation scores
scores = cross_val_score(lin_reg, X, y, cv=kf)

#print scores
print('Scores = ', scores)
print('Mean absolute error = ', np.mean(np.absolute(scores)))
print('RMSE = ', np.sqrt(np.mean(np.absolute(scores))))


#############################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

lin_reg.fit(X_train, y_train)

predictions = lin_reg.predict(X_test)
for i in range(len(predictions)):
    print('AHA was ', y_test[i], ' predicted ', predictions[i])

# Plot the data points
plt.scatter(X, y)

# Plot the regression line
plt.plot(X, lin_reg.predict(X), color='red')

# Add labels and show the plot
plt.xlabel('healthy_percentage')
plt.ylabel('AHA')
plt.show()
'''