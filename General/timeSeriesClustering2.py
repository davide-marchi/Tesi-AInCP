import pandas as pd
from sklearn.cluster import KMeans
from sktime.utils.load_data import load_from_tsfile_to_dataframe

# Set the number of clusters
n_clusters = 3

# Load the first time series data from a CSV file into a pandas DataFrame
df1 = pd.read_csv("timeseries1.csv")

# Load the second time series data from a CSV file into a pandas DataFrame
df2 = pd.read_csv("timeseries2.csv")

# Load the third time series data from a CSV file into a pandas DataFrame
df3 = pd.read_csv("timeseries3.csv")

# Combine all the time series data into a single pandas DataFrame
df = pd.concat([df1, df2, df3])

# Convert the pandas DataFrame into a sktime dataframe
sktime_df = load_from_tsfile_to_dataframe(df)

# Extract the time series data from the sktime dataframe
X = sktime_df.iloc[:, :-1]

# Fit the KMeans model to the time series data
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

# Predict the cluster labels for each time series
y_pred = kmeans.predict(X)

# Print the cluster labels for each time series
print(y_pred)




import matplotlib.pyplot as plt

# Extract the time and value columns from the pandas DataFrame
time = df['time'].values
value = df['value'].values

# Create a scatter plot of the time series data, colored by cluster label
plt.scatter(time, value, c=y_pred)

# Add a legend and show the plot
plt.legend()
plt.show()