import torch
import torch.nn as nn

class TorcsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=5, dropout=0.2):
        """
        LSTM model for TORCS driving with enhanced brake prediction
        
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden layer
            num_layers: Number of LSTM layers
            output_size: Number of output values to predict
            dropout: Dropout probability
        """
        super(TorcsLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate output branches for different control outputs
        self.steer_branch = nn.Linear(64, 1)
        self.accel_branch = nn.Linear(64, 1)
        
        # Enhanced brake branch with more capacity
        self.brake_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Brake should be between 0 and 1
        )
        
        self.gear_branch = nn.Linear(64, 1)
        self.reverse_branch = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output from the last time step
        out = out[:, -1, :]
        
        # Extract shared features
        shared_features = self.shared_fc(out)
        
        # Get individual outputs from each branch
        steer = self.steer_branch(shared_features)
        accel = self.accel_branch(shared_features)
        brake = self.brake_branch(shared_features)
        gear = self.gear_branch(shared_features)
        reverse = self.reverse_branch(shared_features)
        
        # Concatenate outputs
        outputs = torch.cat([steer, accel, brake, gear, reverse], dim=1)
        
        return outputs