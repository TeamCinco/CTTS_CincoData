import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
import glob
import os

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_and_process_data(folder_path):
    # Get list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Initialize empty list to store dataframes
    all_data = []
    
    # Read and process each CSV file
    for file in csv_files:
        df = pd.read_csv(file)
        
        # Convert ts_event to datetime
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        
        # Keep only ts_event and price columns
        df = df[['ts_event', 'price']]
        
        all_data.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('ts_event')
    
    # Resample to 1-minute intervals
    df_1min = combined_df.set_index('ts_event').resample('1T')['price'].last().dropna()
    
    return df_1min.reset_index()

class CNN(nn.Module):
    def __init__(self, input_channels=1, kernel_size=16, stride=4, embed_dim=128):  # Reduced stride
        super(CNN, self).__init__()
        
        # Calculate output size after first conv: ((W - K)/S) + 1
        # With W=80, K=16, S=4: ((80-16)/4) + 1 = 17
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Second conv with smaller kernel size and stride
        # Input size is 17, so use kernel_size=3, stride=2
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, stride=2),
            nn.BatchNorm1d(embed_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Calculate flattened size: ((17-3)/2) + 1 = 8 timesteps * (embed_dim*2) channels
        self.flatten_size = 8 * (embed_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def create_sequences(data, seq_length=80, flat_threshold=0.0001):
    """Create sequences for training/testing with specified threshold"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequence = data[i:(i + seq_length)]
        current_price = data[i + seq_length - 1]
        next_price = data[i + seq_length]
        
        price_change = (next_price - current_price) / current_price
        
        if price_change > flat_threshold:
            target = 0  # up
        elif price_change < -flat_threshold:
            target = 1  # down
        else:
            target = 2  # flat
        
        sequences.append(sequence)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, original_prices, timestamps):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.original_prices = original_prices
        self.timestamps = timestamps
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(folder_path):
    # Load and process data
    print("Loading and processing data...")
    processed_df = load_and_process_data(folder_path)
    print(f"Loaded {len(processed_df)} minutes of data")
    
    # Prepare the data
    price_data = processed_df['price'].values
    timestamps_data = processed_df['ts_event'].values
    
    # Split the data chronologically (80% train, 20% test)
    split_idx = int(len(price_data) * 0.8)
    train_data = price_data[:split_idx]
    test_data = price_data[split_idx:]
    
    train_timestamps = timestamps_data[:split_idx]
    test_timestamps = timestamps_data[split_idx:]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Scale the data
    train_scaler = MinMaxScaler()
    test_scaler = MinMaxScaler()
    
    train_scaled = train_scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_scaled = test_scaler.fit_transform(test_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    print("Creating sequences...")
    X_train, y_train = create_sequences(train_scaled)
    X_test, y_test = create_sequences(test_scaled)
    
    # Reshape sequences for 1D CNN
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # Align timestamps and prices with sequences
    # Note: sequence creation reduces the number of samples
    train_timestamps = train_timestamps[:-1][:len(X_train)]
    test_timestamps = test_timestamps[:-1][:len(X_test)]
    train_prices = train_data[:-1][:len(X_train)]
    test_prices = test_data[:-1][:len(X_test)]
    
    train_dataset = StockDataset(X_train, y_train, train_prices, train_timestamps)
    test_dataset = StockDataset(X_test, y_test, test_prices, test_timestamps)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        if (epoch + 1) % 10 == 0:
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_probs.extend(max_probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    standard_accuracy = 100 * correct / total
    
    # Calculate thresholded accuracy (75th percentile threshold as per paper)
    threshold = np.percentile(all_probs, 75)
    high_conf_indices = np.where(np.array(all_probs) >= threshold)[0]
    
    thresholded_correct = sum(np.array(all_preds)[high_conf_indices] == np.array(all_targets)[high_conf_indices])
    thresholded_accuracy = 100 * thresholded_correct / len(high_conf_indices)
    
    return standard_accuracy, thresholded_accuracy