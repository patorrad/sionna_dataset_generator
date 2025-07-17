import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import wandb
import yaml


def login_wandb(name="None", config=None):
    with open("/home/paolo/Documents/keys.yaml", "r") as file:
        key_data = yaml.safe_load(file) 
    wandb.login(key=key_data["wandb"]) 

    wandb.init(
        project="NASA_DCGR", 
        name=name,
        config=config
    )

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, predictions):
        self.data = data_tensor
        self.predictions = predictions

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]                 # shape (21, 10)
        input_seq = seq[:seq.shape[0] - self.predictions]                 # shape (18, 10)
        target_values = seq[seq.shape[0] - self.predictions:]          # shape (3,) — 3 future values of feature[0]
        return input_seq, target_values
    
class LSTMMultiStep(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=1, num_pred=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_pred)  # Predict 3 future values (1 feature)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]                 # shape: (batch, hidden_size)
        output = self.fc(last_hidden)        # shape: (batch, 3)
        output = self.activation(output)
        return output

# ----- Load Data -----
# data = np.load('data_small_mesh0_2-short_all_depth_2_tx_height_25_250dBm.npy')
data = np.load('data_small_mesh0_2_all_depth_2_tx_height_25_250dBm.npy')
print(f"Loaded dataset shape: {data.shape}")
data_input = data[:, :, :11].reshape(data.shape[0], data.shape[1], 11)

# trajectories = np.load('trajectories_lunar_mesh_ex.npy')
# # (996, 100, 3)
# trajectories = trajectories[:, ::5, :]
# print(f"Trajectories: {trajectories.shape}")
# total_traj = trajectories.shape[0]
# seq_len = 20
# data = np.empty(shape=(0, seq_len, 10 + seq_len * 4 *52))
# print(f"Initial dataset shape: {data.shape}")

# data = np.load("data_lunar_mesh_ex.npy")
# print(f"Loaded data shape: {data.shape}")
# data = data.reshape(total_traj, seq_len - 1, 10 + seq_len * 4 *52)
# data = data[:,:,:10]

num_pred = 10

# Wrap your tensor in the dataset
dataset = TrajectoryDataset(torch.tensor(data, dtype=torch.float32), num_pred)

# Split
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size_train = 100
batch_size_test = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# ----- Instantiate Models -----
latent_dim = 16 # cnn output dimension for CSI embedding
input_features = 1  # first 8 dims of data_input
hidden_size = 128  # default LSTM hidden size
model = LSTMMultiStep(input_size=input_features, num_pred=num_pred)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Wandab configuration
wandb_active = True
num_epochs = 100
name = "RNN_Training_RSSI"
config = {
            "input_size": input_features,  # 8 features + latent CSI embedding
            "num_pred": num_pred,
            "hidden_size": hidden_size,  # default 128
            "latent_dim": latent_dim,
            # "cnn_output_dim": latent_dim,
            # "cnn_kernel_size": kernel_size,
            # "cnn_num_filters": None,  # output channels in CNN
            # "cnn_pool_size": 1,  # adaptive pooling to 1x1
            "batch_size_train": batch_size_train,
            "batch_size_test": batch_size_test,
            "epochs": num_epochs,
            "lr": optimizer.defaults['lr'],
        }
if wandb_active:
    login_wandb(name, config)

# ----- Training Loop -----
for epoch in range(num_epochs):
    for input_seq, target_val in train_loader:
        input_seq = input_seq[:, :, :input_features]
        output = model(input_seq)
    
        # Losses
        loss = criterion(output, target_val[:,:,0])         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Log training loss
    print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")
    if wandb_active: wandb.log({"train_loss": loss.item(), "epoch": epoch})

    # Optional test loop
    rssi_ground_truth = []
    rssi_predicted = []
    pos = []
    rssi_seq = []
    model.eval()
    with torch.no_grad():
        test_losses = []
        for input_seq, target_val in test_loader:
            # pos.append(input_seq[:, -1, 7:])
            pos.append(target_val[:,:,input_features:input_features + 3])
            input_seq = input_seq[:, :, :input_features]
            test_output = model(input_seq)
            rssi_ground_truth.append(target_val[:,:,0].numpy())
            rssi_seq.append(np.concatenate([input_seq.numpy()[:, :, 0], target_val[:,:,0].numpy()], axis=1))
            rssi_predicted.append(test_output.numpy())
            test_loss = criterion(test_output, target_val[:,:,0])
            test_losses.append(test_loss.item())
        avg_test_loss = sum(test_losses) / len(test_losses)

    print(f"[Test] Average Loss: {avg_test_loss:.4f}")
    if wandb_active: wandb.log({"test_loss": avg_test_loss, "epoch": epoch})
    model.train()

# torch.save(model.state_dict(), "NASA_DCGR.pt")

print(len(rssi_seq))
rssi_seq_start = np.array(rssi_seq[0])
print(rssi_seq_start[:,:9].shape)

# Naive prediction
naive_pred = rssi_seq_start[:, 8:9].repeat(10, 1)
print(naive_pred.shape)

# Linear regression
x_fit = np.arange(9)
x_pred = np.arange(9, 19)

# Fit degree-1 polynomial (line)
coeffs = np.polyfit(x_fit, rssi_seq_start[:, :9].T, deg=1)  # shape (2, 995)

# Evaluate line at x_pred
x_pred_powers = np.vstack([x_pred, np.ones_like(x_pred)])  # shape (2, 10)
linear_pred = coeffs.T @ x_pred_powers  # shape (995, 10)
print(linear_pred.shape)

# Cubic spline
from scipy.interpolate import CubicSpline
spline_preds = []

for seq in rssi_seq_start[:, :9]:
    cs = CubicSpline(x_fit, seq[:9], extrapolate=True)
    pred = cs(x_pred)
    spline_preds.append(pred)

spline_pred = np.stack(spline_preds)  # shape (995, 10)
print(spline_pred.shape)

import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Compute MSE (Lower is better)
mse = mean_squared_error(rssi_ground_truth[0], rssi_predicted[0])
mae = mean_absolute_error(rssi_ground_truth[0], rssi_predicted[0])

print(f"RNN Predictions")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

mse = mean_squared_error(rssi_ground_truth[0], linear_pred)
mae = mean_absolute_error(rssi_ground_truth[0], linear_pred)
print(f"Linear Regression Predictions")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

num_points = pos[0].shape[0] * pos[0].shape[1]
distance = np.linalg.norm(pos[0].reshape(num_points, 3), axis=1)
rssi_ground_truth_flat = np.array(rssi_ground_truth[0]).flatten()

# Plotting the ground truth vs predicted RSSI values
plt.figure(figsize=(12, 6))
plt.scatter(distance, rssi_ground_truth_flat, label='Ground Truth RSSI', color='blue', alpha=0.5)
plt.scatter(distance, np.array(rssi_predicted[0]).flatten(), label='Predicted RSSI', color='red', alpha=0.5)
plt.scatter(distance, np.array(linear_pred).flatten(), label='Linear Regression RSSI', color='green', alpha=0.5)
plt.title('Ground Truth vs Predicted RSSI')
plt.xlabel('Distance from Origin (m)')
plt.ylabel('Normalized RSSI')
plt.legend()
plt.grid(True)
# plt.show()
# Log the plot to wandb
wandb.log({"RSSI_vs_Distance": wandb.Image(plt)})
# Optional: close the plot to avoid memory leaks if inside a loop
plt.close()

# # Plotting the error distribution
# plt.figure(figsize=(12, 6))
# errors = np.array(rssi_ground_truth[0]).flatten() - np.array(rssi_predicted[0]).flatten()
# plt.hist(errors, bins=50, color='purple', alpha=0.7)
# errors_reg = np.array(rssi_ground_truth[0]).flatten() - linear_pred.flatten()
# plt.hist(errors_reg, bins=50, color='green', alpha=0.5)
# plt.title('Error Distribution (Ground Truth - Predicted RSSI)')
# plt.xlabel('Error in Normalized RSSI')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# Flatten and compute errors
errors_rnn = np.array(rssi_ground_truth[0]).flatten() - np.array(rssi_predicted[0]).flatten()
errors_linear = np.array(rssi_ground_truth[0]).flatten() - linear_pred.flatten()
# 1. Find shared min/max
combined = np.concatenate([errors_rnn, errors_linear])
bin_edges = np.linspace(combined.min(), combined.max(), 51)  # 50 bins
# 2. Plot both with same bins
plt.figure(figsize=(12, 6))
plt.hist(errors_rnn, bins=bin_edges, color='purple', alpha=0.6, label='RNN Error')
plt.hist(errors_linear, bins=bin_edges, color='green', alpha=0.6, label='Linear Fit Error')
plt.title('Error Distribution (Ground Truth - Predicted RSSI)')
plt.xlabel('Error in Normalized RSSI')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
# plt.show()
# Log the plot to wandb
wandb.log({"Error Distribution (Ground Truth - Predicted RSSI)": wandb.Image(plt)})
# Optional: close the plot to avoid memory leaks if inside a loop
plt.close()


# RNN Predictions
# Mean Squared Error (MSE): 0.0729
# Mean Absolute Error (MAE): 0.1898
# Linear Regression Predictions
# Mean Squared Error (MSE): 0.2488
# Mean Absolute Error (MAE): 0.2765

# RNN Predictions
# Mean Squared Error (MSE): 0.0199
# Mean Absolute Error (MAE): 0.0687
# Linear Regression Predictions
# Mean Squared Error (MSE): 0.0542
# Mean Absolute Error (MAE): 0.0781