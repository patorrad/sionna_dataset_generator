import torch
import torch.nn as nn
import pandas as pd
import csv
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RNNModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=8, num_layers=1, output_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.rnn(x, (h0, c0))
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])  
        return out

# Load dataset
dataset = []
num_prev_steps = 15
with open('dataset.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if present

    for row in csv_reader:
        rssi = float(row[0])  # RSSI normalized value
        angles = ast.literal_eval(row[1])  # Convert string to list
        angle_profiles = ast.literal_eval(row[2])  # Convert string to list
        prev_rssi = ast.literal_eval(row[3])  # Convert string to list

        dataset.append([[rssi] + prev_rssi[num_prev_steps:], angles[:num_prev_steps], angle_profiles[:num_prev_steps], prev_rssi[:num_prev_steps]])

# Convert to DataFrame
df = pd.DataFrame(dataset, columns=['rssi_normalized', 'angles', 'angle_profiles', 'prev_rssi'])
print(df)

# Train-Test Split
X = df[['angles', 'angle_profiles', 'prev_rssi']].values
y = df['rssi_normalized'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert to tensors
def prepare_input(data):
    """ Converts list of angles, profiles, and prev_rssi to a single tensor of shape (sequence_length=18, input_size=7) """
    sequences = []
    for angles, profiles, prev_rssi in data:
        sequence = []
        for i in range(len(angles)):  # Loop over 18 steps
            step_features = angles[i] + profiles[i] + [prev_rssi[i]]  # Add previous RSSI as a feature
            sequence.append(step_features)
        sequences.append(sequence)
    
    return torch.tensor(sequences, dtype=torch.float32)

X_train = prepare_input(X_train)
X_test = prepare_input(X_test)

# Convert y_train from DataFrame to a list of lists
y_train = y_train.tolist()  # Convert pandas DataFrame to a list of lists
y_test = y_test.tolist()

# Convert to a properly shaped NumPy array
y_train = np.array(y_train, dtype=np.float32)  # Ensures a 2D shape (batch_size, 4)
y_test = np.array(y_test, dtype=np.float32)

# Convert to PyTorch tensors
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Reshape X for LSTM: (batch_size, sequence_length=18, input_size=7)
X_train = X_train.view(-1, num_prev_steps, 7)
X_test = X_test.view(-1, num_prev_steps, 7)

# Initialize Model
torch.manual_seed(41)
model = RNNModel(input_size=7)  # Adjust input size

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 100
for i in range(1, epochs+1):
    # Forward pass
    y_pred = model(X_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)


    if i % 10 == 0:
        print(f'Epoch: {i}, Loss: {loss.item()}')
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

average_diff = (y_pred - y_train)/y_train * 100
average_diff_np = average_diff.detach().numpy()
print(f"Percentage Difference: {average_diff_np.mean()}%")