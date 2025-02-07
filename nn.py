import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from util import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MovieDataset(Dataset):
    def __init__(self, X, y):
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(X_scaled)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MovieNN(nn.Module):
    def __init__(self, input_size):
        super(MovieNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_nn(save_path: str, title: str, epochs=50, batch_size=16, learning_rate=0.001):
    # Load and prepare data
    X_train, y_train = load_dataset('train_data.csv', include_genres=False)
    X_test, y_test = load_dataset('test_data.csv', include_genres=False)
    
    # Create datasets and dataloaders
    train_dataset = MovieDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MovieDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = MovieNN(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    for epoch in range(epochs):
        # Training
        model.train()
        train_epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}')

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            all_preds.extend(outputs.numpy())
            all_targets.extend(batch_y.numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_preds)
    y_test = np.array(all_targets)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    logging.info(f'Test MSE: {mse:.4f}')
    logging.info(f'Test R²: {r2:.4f}')
    
    # Plot results
    plot_nn_results(y_test, y_pred, train_losses, [], save_path, title)
    
    return model, mse, r2

def plot_nn_results(y_test, y_pred, train_losses, val_losses, save_path, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Add main title
    fig.suptitle(title, fontsize=16, y=1.05)
    
    # Plot 1: Actual vs Predicted
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted')
    
    # Plot 2: Residuals
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    
    # Plot 3: Training Loss
    ax3.plot(train_losses, label='Training Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_nn(
        save_path='nn_results.png',
        title="Neural Network Regression",
        epochs=500,
        batch_size=16,
        learning_rate=0.001
    )
