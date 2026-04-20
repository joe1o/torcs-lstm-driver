import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from data_processor import load_and_preprocess_data, save_scalers
from model import TorcsLSTM

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=30, save_path='.'):
    """
    Train the LSTM model with weighted loss to emphasize Brake prediction
    
    Args:
        model: PyTorch model
        dataloaders: Dictionary with train and val dataloaders
        criterion: Loss function (MSELoss)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run the training on
        num_epochs: Number of training epochs
        save_path: Path to save the trained model
        
    Returns:
        model: Trained model
        history: Training history
    """
    since = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_loss = float('inf')
    
    # Define weights for each output: [Steer, Accel, Brake, Gear, ReverseMode]
    weights = torch.tensor([1.0, 1.0, 10.0, 1.0, 1.0]).to(device)  # Increased weight for Brake to 10.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            
            # Iterate over data
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Compute weighted MSE loss
                    # outputs and targets shape: (batch_size, 5)
                    # Compute MSE for each output dimension separately
                    mse_losses = (outputs - targets) ** 2  # (batch_size, 5)
                    mse_losses = mse_losses.mean(dim=0)  # (5,)
                    weighted_loss = (weights * mse_losses).sum()  # Scalar
                    
                    loss = weighted_loss
                
                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Save loss for history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)
                
                # Update learning rate
                scheduler.step(epoch_loss)
                print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Save best model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, os.path.join(save_path, 'torcs_model_best.pth'))
                    print(f'Saved best model with loss {best_loss:.4f}')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': best_loss,
    }, os.path.join(save_path, 'torcs_model_final.pth'))
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train TORCS LSTM model')
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\abdul\Desktop\AI project\pyScrcClient-master\dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--save_path', type=str, default='.',
                        help='Path to save the trained model')
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Sequence length for LSTM input')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150,  # Increased epochs as per your output
                        help='Number of epochs')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    dataloaders, scalers = load_and_preprocess_data(args.dataset_path, args.seq_length)
    
    # Save scalers for inference
    save_scalers(scalers, args.save_path)
    
    # Get input size from the first batch of training data
    sample_batch, _ = next(iter(dataloaders['train']))
    input_size = sample_batch.shape[2]
    
    # Create model
    model = TorcsLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=5,  # Steer, Accel, Brake, Gear, ReverseMode
        dropout=args.dropout
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction='none')  # We'll compute weighted loss manually
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_path=args.save_path
    )
    
if __name__ == '__main__':
    main()