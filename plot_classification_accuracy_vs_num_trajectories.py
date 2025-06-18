import matplotlib.pyplot as plt
import numpy as np

# Data
trajectories = np.array([100, 500, 1000, 2000, 5000, 10000])
accuracy = np.array([72.4, 80.1, 85.7, 88.9, 91.3, 93.5])
std_dev = np.array([2.5, 1.8, 1.2, 1.0, 0.7, 0.5])  # example standard deviations

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(trajectories, accuracy, yerr=std_dev, fmt='o-', capsize=5, elinewidth=1.2, markerfacecolor='blue')

# Labels and title
plt.xlabel("Number of Trained Trajectories")
plt.ylabel("Test Accuracy (%)")
plt.title("Classification Test Accuracy vs. Number of Trained Trajectories")
plt.grid(True)
plt.xscale('log')  # optional: makes it easier to visualize large range
plt.xticks(trajectories, labels=[str(n) for n in trajectories])  # optional for nicer ticks

# Show the plot
plt.tight_layout()
plt.show()
