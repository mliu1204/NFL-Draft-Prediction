import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import matplotlib.pyplot as plt

from util import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ultra Basic Linear Regression w/ Genres
def linear_regression(include_genres: bool, save_path: str, title: str):
    x_train, y_train = load_dataset('train_data.csv', include_genres=include_genres)
    x_test, y_test = load_dataset('test_data.csv', include_genres=include_genres)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Calculate and print accuracy metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f'Mean Squared Error: {mse:.2f}')
    logging.info(f'R² Score: {r2:.2f}')
    
    plot_regression_results(y_test, y_pred, save_path, title)
    
    return model, mse, r2

def plot_regression_results(y_test, y_pred, save_path=None, title="Untitled"):
    """
    Plot actual vs predicted values and residuals
    Args:
        y_test: True values
        y_pred: Predicted values
        save_path: Optional path to save the plot
        title: Main title for the plot (default: "Regression Analysis")
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Add main title to the figure
    fig.suptitle(title, fontsize=16, y=1.05)
    
    # Actual vs Predicted plot
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted')
    
    # Residuals plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

linear_regression(include_genres=False, save_path='lr_no_genres.png', title="LinReg w/ NO Genres")
linear_regression(include_genres=True, save_path='lr_with_genres.png', title="LinReg w/ Genres")
