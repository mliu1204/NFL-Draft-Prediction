import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

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

def plot_feature_importance(model, feature_names, save_path=None, title="Feature Importance"):
    """
    Create a heatmap of feature coefficients/importance
    Args:
        model: Trained model (Ridge or Lasso)
        feature_names: List of feature names
        save_path: Optional path to save the plot
        title: Title for the plot
    """
    # Get coefficients and create a DataFrame
    coefficients = model.coef_
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create heatmap
    sns.heatmap(
        coef_df.set_index('Feature')['Coefficient'].to_frame().T,
        cmap='RdBu',
        center=0,
        annot=True,
        fmt='.3f'
    )
    
    plt.title(title)
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def ridge_regression(include_genres: bool, save_path: str, title: str, alpha=1.0):
    x_train, y_train = load_dataset('train_data.csv', include_genres=include_genres)
    x_test, y_test = load_dataset('test_data.csv', include_genres=include_genres)
    
    # Get feature names
    if hasattr(x_train, 'columns'):
        feature_names = x_train.columns
    else:
        feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
    
    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Initialize and train the model
    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test_scaled)
    
    # Calculate and print accuracy metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f'Ridge Regression Results:')
    logging.info(f'Alpha: {alpha}')
    logging.info(f'Mean Squared Error: {mse:.2f}')
    logging.info(f'R² Score: {r2:.2f}')
    
    # Plot regression results
    plot_regression_results(y_test, y_pred, save_path, title)
    
    # Plot feature importance
    feature_importance_path = save_path.replace('.png', '_features.png')
    plot_feature_importance(
        model, 
        feature_names, 
        feature_importance_path, 
        f"Feature Importance - Ridge (α={alpha})"
    )
    
    return model, mse, r2

def lasso_regression(include_genres: bool, save_path: str, title: str, alpha=1.0):
    x_train, y_train = load_dataset('train_data.csv', include_genres=include_genres)
    x_test, y_test = load_dataset('test_data.csv', include_genres=include_genres)
    
    # Get feature names
    if hasattr(x_train, 'columns'):
        feature_names = x_train.columns
    else:
        feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
    
    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Initialize and train the model
    model = Lasso(alpha=alpha)
    model.fit(x_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test_scaled)
    
    # Calculate and print accuracy metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f'Lasso Regression Results:')
    logging.info(f'Alpha: {alpha}')
    logging.info(f'Mean Squared Error: {mse:.2f}')
    logging.info(f'R² Score: {r2:.2f}')
    
    # Print non-zero coefficients
    non_zero_features = [(name, coef) for name, coef in zip(feature_names, model.coef_) if coef != 0]
    logging.info("Non-zero coefficients:")
    for name, coef in non_zero_features:
        logging.info(f"{name}: {coef:.4f}")
    
    # Plot regression results
    plot_regression_results(y_test, y_pred, save_path, title)
    
    # Plot feature importance
    feature_importance_path = save_path.replace('.png', '_features.png')
    plot_feature_importance(
        model, 
        feature_names, 
        feature_importance_path, 
        f"Feature Importance - Lasso (α={alpha})"
    )
    
    return model, mse, r2

# linear_regression(include_genres=False, save_path='lr_no_genres.png', title="LinReg w/ NO Genres")
# linear_regression(include_genres=True, save_path='lr_with_genres.png', title="LinReg w/ Genres")

alphas = [0.1, 1.0, 10.0]
for alpha in alphas:
    ridge_regression(
        include_genres=False, 
        save_path=f'ridge_with_genres_alpha_{alpha}.png',
        title=f"Ridge Regression (α={alpha}) w/ Genres",
        alpha=alpha
    )

