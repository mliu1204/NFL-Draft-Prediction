import pandas as pd
import numpy as np
import ast

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

def get_unique_genres(df):
    # Get all unique genres from the dataset using numpy's unique
    return np.sort(np.unique(np.concatenate(df['genre_names'].values)))

def get_unique_companies(df):
    # Get all unique production companies from the dataset using numpy's unique
    return np.sort(np.unique(np.concatenate(df['company_names'].values)))

def prepare_for_regression(df, target_column='Minority Percent'):
    # Preprocess features
    X = preprocess_movie_data(df)
    
    # Prepare target variable
    y = df[target_column]
    
    return X, y

X, y = prepare_for_regression(pd.read_csv('train_data.csv'))

X.to_csv('output.csv', index=False)
# print(y.head())

