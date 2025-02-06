import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    # Read the CSV file
    data = pd.read_csv('data_combined_percentages.csv')

    # Remove the 'Non-Male Percent' column
    data = data.drop('Non-Male Percent', axis=1)

    # Split into features (X) and target (y)
    X = data.drop('Minority Percent', axis=1)  # Replace 'target_column' with your actual target column name
    y = data['Minority Percent']

    # Create train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # If you want to save the splits to separate CSV files (optional)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)