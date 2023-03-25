import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create the subplots
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

# Plot the data on each subplot
axs[0].plot(x, y1)
axs[1].plot(x, y2)
axs[2].plot(x, y3)

# Save each subplot as a separate image file
axs[0].figure.savefig("subplot_1.png")
axs[1].figure.savefig("subplot_2.png")
axs[2].figure.savefig("subplot_3.png")