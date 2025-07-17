import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ----- CNN Head -----
# class CSIEncoderCNN(nn.Module):
#     def __init__(self, output_dim=32, kernel_size=3):
#         super(CSIEncoderCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(32, output_dim)

#     def forward(self, x):
#         # x: [B*T, 1, 4, 52]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x).view(x.size(0), -1)
#         return self.fc(x)
class CSIEncoderCNN(nn.Module):
    def __init__(self, output_dim=32, kernel_size=3):
        super(CSIEncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 3), padding=(0, 1))   # keeps 1x52
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 3), padding=(0, 1))  # keeps 1x52
        self.bn2 = nn.BatchNorm2d(16)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → [B*T, 16, 1, 1]
        self.fc = nn.Linear(16, output_dim)

    def forward(self, x):
        # x: [B*T, 1, 1, 52]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ----- RNN Body -----
class LSTMMultiStep(nn.Module):
    def __init__(self, input_size, num_pred, hidden_size=128):
        super(LSTMMultiStep, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_pred)

    def forward(self, x):
        # x: [B, T, input_size]
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]  # Use final time step
        return self.fc(last_hidden)  # Predict [B, num_pred]

# ----- Dataset Wrapper -----
class TrajectoryDataset(Dataset):
    def __init__(self, data, num_pred):
        self.data = data
        self.num_pred = num_pred

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-self.num_pred], seq[-self.num_pred:]

# ----- Load Data -----
file_name = 'data_small_mesh0_2_all_depth_2_tx_height_25_250dBm.npy'
data = np.load(file_name)
print(f"Loaded dataset shape: {data.shape}")
data_input = data[:, :, :11].reshape(data.shape[0], data.shape[1], 11)
data_csi = data[:, :, 11:].reshape(data.shape[0], data.shape[1], 208)

# Assumes CSI is real or complex magnitude
csi_mag = np.abs(data_csi)  # or use np.sqrt(real**2 + imag**2)
csi_log = np.log10(csi_mag + 1e-12)  # avoid log(0)
# Normalize to zero mean, unit variance
mean = np.mean(csi_log)
std = np.std(csi_log)
data_csi = (csi_log - mean) / std

num_pred = 10
dataset = TrajectoryDataset(torch.tensor(np.concatenate([data_input, data_csi], axis=-1), dtype=torch.float32), num_pred)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size_train = 100
batch_size_test = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# ----- Instantiate Models -----
latent_dim = 16 # cnn output dimension for CSI embedding
input_features = 8  # first 8 dims of data_input
hidden_size = 128  # default LSTM hidden size
model = LSTMMultiStep(input_size=input_features + latent_dim, num_pred=num_pred, hidden_size=hidden_size).to(device)
kernel_size = 4
cnn_encoder = CSIEncoderCNN(output_dim=latent_dim, kernel_size=kernel_size).to(device)

# ----- Loss and Optimizer -----
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(cnn_encoder.parameters()), lr=1e-3)

# Wandab configuration
wandb_active = True
num_epochs = 100
name = "RNN_CSI_Embedding_Training_1_ant"
config = {
            "input_size": input_features + latent_dim,  # 8 features + latent CSI embedding
            "num_pred": num_pred,
            "hidden_size": hidden_size,  # default 128
            "latent_dim": latent_dim,
            "cnn_output_dim": latent_dim,
            "cnn_kernel_size": kernel_size,
            "cnn_num_filters": file_name,  # output channels in CNN
            "cnn_pool_size": 1,  # adaptive pooling to 1x1
            "batch_size_train": batch_size_train,
            "batch_size_test": batch_size_test,
            "epochs": num_epochs,
            "lr": optimizer.defaults['lr'],
        }
login_wandb(name, config)

# ----- Training Loop -----
for epoch in range(num_epochs):
    model.train()
    cnn_encoder.train()
    for input_seq, target_val in train_loader:
        input_seq = input_seq.to(device)
        target_val = target_val.to(device)

        # Split input into features and CSI
        main_input = input_seq[:, :, :input_features]     # [B, T, 8]
        # csi_input = input_seq[:, :, input_features + 3:]      # [B, T, 208]
        csi_first = input_seq[:, :, :52]  # [B, T, 52]

        # Reshape CSI for CNN: [B*T, 1, 4, 52]
        B, T, _ = csi_first.shape
        # csi_reshaped = csi_input.view(B * T, 1, 4, 52)
        # csi_embedding = cnn_encoder(csi_reshaped)         # [B*T, latent_dim]
        # csi_embedding = csi_embedding.view(B, T, -1)      # [B, T, latent_dim]
        csi_first_reshaped = csi_first.reshape(B * T, 1, 1, 52)  # [B*T, 1, 1, 52]
        csi_embedding = cnn_encoder(csi_first_reshaped)         # [B*T, latent_dim]
        csi_embedding = csi_embedding.view(B, T, -1)      # [B, T, latent_dim]

        # import pdb; pdb.set_trace()
        # Concatenate and feed into LSTM
        lstm_input = torch.cat([main_input, csi_embedding], dim=-1)  # [B, T, 8 + latent_dim]
        output = model(lstm_input)  # [B, num_pred]
        loss = criterion(output, target_val[:, :, 0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")
    # Log training loss
    if wandb_active: wandb.log({"train_loss": loss.item(), "epoch": epoch})

    # Optional test loop
    rssi_ground_truth = []
    rssi_predicted = []
    pos = []
    rssi_seq = []

    model.eval()
    cnn_encoder.eval()  # Important!

    with torch.no_grad():
        test_losses = []
        for input_seq, target_val in test_loader:
            input_seq = input_seq.to(device)
            target_val = target_val.to(device)

            # Split into feature and CSI components
            main_input = input_seq[:, :, :8]             # [B, T, 8]
            csi_input  = input_seq[:, :, 8:216]          # [B, T, 208]

            # Save position (if needed)
            pos.append(target_val[:, :, input_features:input_features + 3].cpu().numpy())

            # Reshape CSI for CNN
            B, T, _ = csi_input.shape
            csi_reshaped = csi_input.reshape(B * T, 1, 4, 52)
            csi_embedding = cnn_encoder(csi_reshaped)         # [B*T, latent]
            csi_embedding = csi_embedding.reshape(B, T, -1)   # [B, T, latent]

            # Concatenate inputs
            lstm_input = torch.cat([main_input, csi_embedding], dim=-1)  # [B, T, 8+latent]

            # Predict
            test_output = model(lstm_input)                  # [B, num_pred]
            test_loss = criterion(test_output, target_val[:, :, 0])
            test_losses.append(test_loss.item())

            # Save outputs
            rssi_ground_truth.append(target_val[:, :, 0].cpu().numpy())
            rssi_predicted.append(test_output.cpu().numpy())

            # For visualization
            rssi_seq.append(
                np.concatenate(
                    [main_input[:, :, 0].cpu().numpy(), target_val[:, :, 0].cpu().numpy()],
                    axis=1
                )
            )

            avg_test_loss = sum(test_losses) / len(test_losses)
            
    print(f"[Test] Average Loss: {avg_test_loss:.4f}")
    if wandb_active: wandb.log({"test_loss": avg_test_loss, "epoch": epoch})

# Optional: Save model
# torch.save({
#     'lstm_model': model.state_dict(),
#     'cnn_encoder': cnn_encoder.state_dict()
# }, 'cnn_rnn_model.pt')

import trimesh
# mesh = trimesh.load_mesh("models/lunar_mesh_ex.ply")
# mesh = trimesh.load_mesh("models/canyon.ply")
mesh = trimesh.load_mesh("models/meshes_512/small_mesh0.ply")
import numpy as np

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
plt.show()
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
plt.show()
# # Plotting the error against distance
# plt.figure(figsize=(12, 6))
# plt.scatter(distance.numpy(), errors_rnn, color='green', alpha=0.5)
# plt.title('Error vs Distance from Origin')
# plt.xlabel('Distance from Origin (m)')
# plt.ylabel('Error in Normalized RSSI (Ground Truth - Predicted)')
# plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
# plt.legend()
# plt.grid(True)
# plt.show()