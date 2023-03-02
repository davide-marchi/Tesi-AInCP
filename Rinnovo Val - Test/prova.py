import numpy as np

# Create two arrays to shuffle
X = np.array([1,3,5,7])
y = np.array([10, 30, 50, 70])

# Generate a permutation index and use it to shuffle both arrays
permutation_idx = np.random.permutation(len(X))
X_shuffled = [X[permutation_idx]]
y_shuffled = y[permutation_idx]

# Print the shuffled arrays
print("X shuffled:", X_shuffled)
print("y shuffled:", y_shuffled)
