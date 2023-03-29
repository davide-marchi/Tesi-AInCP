from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import matplotlib.pyplot as plt

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


predictions_dataframe = pd.read_csv('week_stats/predictions_dataframe.csv', index_col=0)

predictions_dataframe['CPI'] = [float(string.strip('[]')) for string in predictions_dataframe['healthy_percentage']]

X = predictions_dataframe[['CPI']].values
y = metadata['AHA'].values

# Assuming you have your dataset as numpy array X and target y
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error and R-squared score for the predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ",(mse))
print("R-squared score: ",(r2))

# Plot the data points
plt.scatter(X, y)

# Plot the regression line
plt.plot(X, model.predict(X), color='red')

# Add labels and show the plot
plt.xlabel('healthy_percentage')
plt.ylabel('AHA')
plt.show()
plt.close()