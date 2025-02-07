import pandas as pd
import numpy as np
import ast
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# YOU SHOULD CALL THESE FUNCTIONS IN LINEAR.PY

def load_dataset(csv_path, target_column='Minority Percent', include_genres=False):
    """Load dataset from csv file"""
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing values
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len:
        logging.info(f"Dropped {original_len - len(df)} rows with missing values")
    
    X, y = prepare_for_regression(df, target_column, include_genres)
    return X.to_numpy(), y.to_numpy()

# ------------------------------------------------------------------------------------------------

# YOU SHOULDN'T HAVE TO CALL THESE FUNCTIONS IN LINEAR.PY

def preprocess_movie_data(df):
    # Convert string representations of lists to actual lists/dicts
    df['genres'] = df['genres'].apply(ast.literal_eval)
    
    # Extract genre names
    df['genre_names'] = df['genres'].apply(lambda x: [genre['name'] for genre in x])
    
    # Get unique genres
    unique_genres = get_unique_genres(df)
    
    # One-hot encode genres using NumPy operations
    genre_matrix = np.zeros((len(df), len(unique_genres)))
    for i, genres in enumerate(df['genre_names']):
        genre_indices = [np.where(unique_genres == genre)[0][0] for genre in genres]
        genre_matrix[i, genre_indices] = 1
    
    # Convert release_date to datetime and extract useful features
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    
    # Create final feature set
    numeric_features = df[['budget', 'popularity', 'runtime', 'revenue']].to_numpy()
    time_features = df[['release_year', 'release_month']].to_numpy()
    
    # Combine all features using NumPy horizontal stack
    features = np.hstack([numeric_features, genre_matrix, time_features])
    
    # Convert back to DataFrame with proper column names
    feature_columns = (
        list(df[['budget', 'popularity', 'runtime', 'revenue']].columns) +
        [f'genre_{genre}' for genre in unique_genres] +
        ['release_year', 'release_month']
    )
    features = pd.DataFrame(features, columns=feature_columns, index=df.index)
    
    return features

def preprocess_movie_data_no_genres(df):
    """Preprocess movie data without genre information"""
    # Convert release_date to datetime and extract useful features
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    
    # Create final feature set with only numeric and time features
    numeric_features = df[['budget', 'popularity', 'runtime', 'revenue']].to_numpy()
    time_features = df[['release_year', 'release_month']].to_numpy()
    
    # Combine all features using NumPy horizontal stack
    features = np.hstack([numeric_features, time_features])
    
    # Convert back to DataFrame with proper column names
    feature_columns = (
        list(df[['budget', 'popularity', 'runtime', 'revenue']].columns) +
        ['release_year', 'release_month']
    )
    features = pd.DataFrame(features, columns=feature_columns, index=df.index)
    
    return features

def get_unique_genres(df):
    # Get all unique genres from the dataset using numpy's unique
    return np.sort(np.unique(np.concatenate(df['genre_names'].values)))

def get_unique_companies(df):
    # Get all unique production companies from the dataset using numpy's unique
    return np.sort(np.unique(np.concatenate(df['company_names'].values)))

def prepare_for_regression(df, target_column='Minority Percent', include_genres=False):
    # Preprocess features
    if include_genres:
        X = preprocess_movie_data(df)
    else:
        X = preprocess_movie_data_no_genres(df)
    
    # Prepare target variable
    y = df[target_column]
    
    return X, y

def parse_retention_curve(curve_data):
    """
    Parse SVG path retention curve data into x,y coordinates
    Returns: tuple of (times, retention_values)
    """
    # Split the string into commands
    parts = curve_data.strip().split(' ')
    
    times = []
    retention_values = []
    
    i = 0
    while i < len(parts):
        if parts[i] in ['M', 'C']:
            i += 1
            continue
            
        if ',' in parts[i]:
            time, retention = parts[i].split(',')
            times.append(float(time))
            retention_values.append(float(retention))
        i += 1
    
    return np.array(times), np.array(retention_values)

def visualize_retention_curve(curve_data, normalize=False, figsize=(12, 6)):
    """
    Visualize the retention/engagement curve using matplotlib.
    
    Args:
        curve_data (str): SVG path data string
        normalize (bool): If True, show normalized retention
        figsize (tuple): Figure size (width, height)
    """
    # Parse the curve data
    times, values = parse_retention_curve(curve_data)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(times, values, 'b-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time in Video')
    plt.ylabel('Percentage')
    title = 'Video Retention Curve' if normalize else 'Video Engagement Curve'
    plt.title(title)
    
    # Add grid and set y-axis limits
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, max(values) * 1.05)  # Add 5% padding above max value
    
    # Add horizontal line at 100%
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
df = pd.read_csv('youtube.csv')
print(df.loc[1, 'retentionCurve'])
visualize_retention_curve(df.loc[0, 'retentionCurve'])