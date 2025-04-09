import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import wandb
import yaml


def login_wandb():
    with open("/home/paolo/Documents/keys.yaml", "r") as file:
        key_data = yaml.safe_load(file) 
    wandb.login(key=key_data["wandb"]) 

    wandb.init(
        project="NASA_DCG", 
        name="100_down_to_20_RNN_every2",
        config={
            "input_size": 7,
            "seq_len": 18,
            "hidden_size": 64,
            "num_layers": 1,
            "batch_size": 5,
            "epochs": num_epochs,
            "lr": 1e-3
        }
    )

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, predictions):
        self.data = data_tensor
        self.predictions = predictions

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]                 # shape (21, 10)
        # print(seq.shape[0])
        # print(seq.shape[0] - self.predictions)
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

  

if __name__ == "__main__":

    wandb_active = False
    num_epochs = 100
    if wandb_active:
        login_wandb()

    trajectories = np.load('trajectories_lunar_mesh_ex.npy')
    # (996, 100, 3)
    trajectories = trajectories[:, ::5, :]
    print(f"Trajectories: {trajectories.shape}")
    total_traj = trajectories.shape[0]
    seq_len = 20
    data = np.empty(shape=(0, seq_len, 10 + seq_len * 4 *52))
    print(f"Initial dataset shape: {data.shape}")

    data = np.load("data_lunar_mesh_ex.npy")
    print(f"Loaded data shape: {data.shape}")
    data = data.reshape(total_traj, seq_len - 1, 10) 
    num_pred = 10

    # Wrap your tensor in the dataset
    dataset = TrajectoryDataset(torch.tensor(data, dtype=torch.float32), num_pred)

    # Split
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    model = LSTMMultiStep(input_size=7, num_pred=num_pred)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # tx_power = torch.from_numpy(tx.power_dbm.numpy()).to("cuda:0")
    # tx_power = torch.tensor(tx.power_dbm, dtype=torch.float32, device="cuda:0")
    tx_power = torch.tensor(tx.power_dbm.numpy(), dtype=torch.float32, device="cpu")
    # wandb_active = True
    for epoch in range(num_epochs):
        for input_seq, target_val in train_loader:
            input_seq = input_seq[:, :, :7]
            output = model(input_seq)
            ####
            positions = torch.zeros_like(target_val[:,:,7:])  # Initialize positions tensor
            positions[:,:, :] = torch.tensor(tx_position, dtype=torch.float32).unsqueeze(0)  # Broadcast tx_position to match target_val shape
            # print(f"positions: {target_val[:,:,7:].shape}")
            # Add PINNs loss: penalize deviation from the trajectory
            distance_m = torch.linalg.norm(positions - target_val[:,:,7:], dim=2) # meters
            frequency_hz =  5.745e9  # 2.4 GHz
            c = 3e8  # speed of light in m/s
            loss_db = tx_power - (20 * torch.log10(distance_m * frequency_hz) - 147.55)
            loss_db[loss_db < -90] = -90 # Set a floor for RSS values to avoid extreme values
            # Define range
            min_rss = -90
            max_rss = 0
            # Normalize values to 0-1
            normalized_path_loss = (loss_db - min_rss) / (max_rss - min_rss)
            ####
            # Losses
            loss_sim = criterion(output, target_val[:,:,0]) 
            loss_theory = torch.nn.functional.mse_loss(target_val[:,:,0], normalized_path_loss)

            # Weighting
            lambda_sim = 1.0
            lambda_theory = 0.5  # up to you — tune this

            # Total PINN-style loss
            loss = lambda_sim * loss_sim + lambda_theory * loss_theory
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log training loss
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
                pos.append(target_val[:,:,7:])
                input_seq = input_seq[:, :, :7]
                test_output = model(input_seq)
                rssi_ground_truth.append(target_val[:,:,0].numpy())
                rssi_seq.append(np.concatenate([input_seq.numpy()[:, :, 0], target_val[:,:,0].numpy()], axis=1))
                rssi_predicted.append(test_output.numpy())
                test_loss = criterion(test_output, target_val[:,:,0])
                test_losses.append(test_loss.item())
            avg_test_loss = sum(test_losses) / len(test_losses)

        if wandb_active: wandb.log({"test_loss": avg_test_loss, "epoch": epoch})
        model.train()

    torch.save(model.state_dict(), "NASA_DCGR.pt")