import numpy as np
import matplotlib.pyplot as plt

with open("data/experiments/1219/action/24/120.npy", "rb") as f:
    actions = np.load(f, allow_pickle=True).item()

# xyz_rpy_gripper = actions["action"]
raw_action = actions["raw_action"]
xyz_rpy_gripper = np.zeros((raw_action.shape[0], 7))
xyz_rpy_gripper[:, :3]= raw_action[:, :3]
xyz_rpy_gripper[:, 6] = raw_action[:, 9]

# Define labels for each dimension
labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]

# Plot each dimension separately
plt.figure(figsize=(15, 10))
for i in range(7):
    plt.subplot(4, 2, i + 1)  # Create a subplot (4 rows, 2 columns, index i+1)
    plt.plot(xyz_rpy_gripper[:, i], label=labels[i])
    plt.title(labels[i])
    plt.xlabel("Time step")
    plt.ylabel(labels[i])
    plt.grid(True)
    plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

