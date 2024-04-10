import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ncps.torch import LTC
from ncps.wirings import AutoNCP

# Load the CSV log file using pandas
df = pd.read_csv("log/lightning_logs/version_1/metrics.csv")

# Extract the training and validation loss columns
train_losses = df["train_loss"].values

# Plot the loss curves
plt.figure(figsize=(10, 6))
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# --------------

# Redefining vars from main.py
N = 48
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# Target output is a sine with double the frequency of the input signal
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
print("data_x.shape: ", str(data_x.shape))
print("data_y.shape: ", str(data_y.shape))
data_x = torch.Tensor(data_x)
data_y = torch.Tensor(data_y)

out_features = 1
in_features = 2

wiring = AutoNCP(16, out_features)  # 16 units, 1 motor neuron

# --------------

# Create an instance of the LTC model
ltc_model = LTC(in_features, wiring, batch_first=True)

# Load the trained model state dictionary
state_dict = torch.load('trained_model2.pth')

# Load the state dictionary into the model
ltc_model.load_state_dict(state_dict)

# How does the trained model now fit to the sinusoidal function?
sns.set()
with torch.no_grad():
    prediction = ltc_model(data_x)[0].numpy()
plt.figure(figsize=(6, 4))
plt.plot(data_y[0, :, 0], label="Target output")
plt.plot(prediction[0, :, 0], label="NCP output")
plt.ylim((-1, 1))
plt.title("After training")
plt.legend(loc="upper right")
plt.show()
