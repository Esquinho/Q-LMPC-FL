#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

save_path = os.path.expanduser('~/shared_ws/vasco_ws/src/Q-LMPC-FL/Laboratorio/Updating strategies/Training/fuzzy_samples.csv')
data = pd.read_csv(save_path)

# Step 2: Preprocess the data
# Replace 'target_column' with the name of your target column
X = data.iloc[:, :4].values  # This will take the first 4 columns for the input data
y = data.iloc[:, 4].values
print("X shape:", X.shape)  # Should be (number_of_samples, number_of_features)
print("y shape:", y.shape)  # Should be (number_of_samples,)

# Normalize the features
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

mean_std = {
    "mean": scaler.mean_,
    "std": scaler.scale_
}
# Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Step 3: Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output size is 1 for regression/binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression; use sigmoid for binary classification
        return x

model = NeuralNetwork()

# Step 4: Define Loss and Optimizer
loss_fn = nn.BCEWithLogitsLoss() if len(np.unique(y)) == 2 else nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the Model
num_epochs = 50
batch_size = 64

# Convert tensors to DataLoader for batching
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

losses = []
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X).squeeze()
        loss = loss_fn(y_pred, batch_y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# Step 6: Evaluate the Model
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    test_loss = loss_fn(y_pred, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# Step 7: Plot Performance
# Plot training loss
plt.plot(losses, label='Training Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# For regression: Predicted vs Actual
if len(np.unique(y)) != 2:
    y_pred = y_pred.numpy()
    y_test_np = y_test_tensor.numpy()
    plt.scatter(y_test_np, y_pred, alpha=0.6)
    plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Values')
    plt.grid()
    plt.show()

# Absolute path to the model weights file
save_path = os.path.expanduser('~/shared_ws/vasco_ws/src/Q-LMPC-FL/Laboratorio/Updating strategies/Training/fuzzy_samples.csv')

model_weights_path = os.path.expanduser('~/shared_ws/vasco_ws/src/Q-LMPC-FL/Laboratorio/Updating strategies/Training/fuzzy_NN_parameters_2.pth')
# Save the model weights
torch.save({
    "model_state_dict": model.state_dict(),
    "normalization": mean_std
}, model_weights_path)