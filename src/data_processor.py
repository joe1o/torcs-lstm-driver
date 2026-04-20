import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TorcsDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_and_preprocess_data(dataset_path, seq_length=10):
    """
    Load and preprocess TORCS dataset
    
    Args:
        dataset_path: Path to the folder containing CSV files
        seq_length: Length of sequence for LSTM input
        
    Returns:
        dataloaders: Dictionary with train and val dataloaders
        scalers: Dictionary with feature and target scalers
    """
    all_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    print(f"Found {len(all_files)} CSV files in {dataset_path}")
    
    # Combine all CSV files
    dfs = []
    for file in all_files:
        file_path = os.path.join(dataset_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Drop rows with NaN values if any
    combined_df = combined_df.dropna()
    
    # Log Brake distribution
    print("Brake distribution statistics:")
    print(f"Mean Brake: {combined_df['Brake'].mean():.4f}")
    print(f"Std Brake: {combined_df['Brake'].std():.4f}")
    print(f"Min Brake: {combined_df['Brake'].min():.4f}")
    print(f"Max Brake: {combined_df['Brake'].max():.4f}")
    print(f"Percentage of non-zero Brake values: {(combined_df['Brake'] > 0).mean() * 100:.2f}%")
    
    # Select feature columns
    feature_columns = ['SpeedX', 'SpeedY', 'SpeedZ', 'Angle', 'TrackPos', 'RPM'] + \
                     [f'track[{i}]' for i in range(1, 20)] + \
                     [f'opponent[{i}]' for i in range(1, 37)]
    
    # Select target columns (what we want to predict)
    target_columns = ['Steer', 'Accel', 'Brake', 'Gear', 'ReverseMode']
    
    # Normalize features
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    features = feature_scaler.fit_transform(combined_df[feature_columns])
    targets = target_scaler.fit_transform(combined_df[target_columns])
    
    # Create sequences for LSTM
    X, y = create_sequences(features, targets, seq_length)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TorcsDataset(X_train, y_train)
    val_dataset = TorcsDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    scalers = {
        'features': feature_scaler,
        'targets': target_scaler,
        'feature_columns': feature_columns,
        'target_columns': target_columns
    }
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    return dataloaders, scalers

def create_sequences(features, targets, seq_length):
    """
    Create sequences for LSTM input
    
    Args:
        features: Normalized feature array
        targets: Normalized target array
        seq_length: Length of sequence
        
    Returns:
        X: Sequence input features
        y: Target values
    """
    X = []
    y = []
    
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])
    
    return np.array(X), np.array(y)

def save_scalers(scalers, save_path='.'):
    """Save the scalers for later use in inference"""
    import pickle
    
    with open(os.path.join(save_path, 'torcs_scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"Scalers saved to {os.path.join(save_path, 'torcs_scalers.pkl')}")

def load_scalers(file_path):
    """Load the saved scalers"""
    import pickle
    
    with open(file_path, 'rb') as f:
        scalers = pickle.load(f)
    
    return scalers