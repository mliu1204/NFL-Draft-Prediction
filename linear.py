import pandas as pd
from prepare_data import prepare_data
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def example():
    # Returns data and masks. Example below is how to separate data with masks.
    data, train_set, test_set, holdout_set = prepare_data()
    train_data = data[train_set]
    test_data = data[test_set]
    holdout_data = data[holdout_set]

    print(train_data.head())
    print(test_data.head())
    print(holdout_data.head())
    
    # The label is the pick. In the article that we are following, they make the label whether the player
    # was picked in the first round or not.
    # We can do this by using the pick column.
    # Pick 257 means undrafted, as there are 256 picks in the first round.
    train_data['label'] = train_data['pick'].apply(lambda x: 1 if x <= 32 else 0)
    test_data['label'] = test_data['pick'].apply(lambda x: 1 if x <= 32 else 0)
    holdout_data['label'] = holdout_data['pick'].apply(lambda x: 1 if x <= 32 else 0)

def create_feature_matrix(data):
    # Define all possible positions that could appear in any dataset
    all_positions = ['C', 'CB', 'DB', 'DE', 'DL', 'DT', 'FB', 'G', 'ILB', 'K', 
                    'LB', 'LS', 'NT', 'OG', 'OL', 'OLB', 'OT', 'P', 'QB', 'RB', 
                    'S', 'T', 'TE', 'WR']
    
    # Create dummy variables for position with all possible positions
    pos_dummies = pd.get_dummies(data['pos'], prefix='pos')
    
    # Add missing position columns with zeros
    for pos in all_positions:
        col = f'pos_{pos}'
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0
    
    # Ensure consistent column ordering for positions
    pos_columns = [f'pos_{pos}' for pos in all_positions]
    pos_dummies = pos_dummies[pos_columns]
    
    # List of numeric features
    numeric_features = [
        'year', 'age', 'height', 'weight', 'forty', 'bench', 'vertical',
        'threecone', 'broad', 'shuttle', 'games', 'seasons',
        'pass_cmp', 'pass_att', 'pass_yds', 'pass_int', 'pass_td',
        'rec_yds', 'rec_td', 'rec', 'rush_att', 'rush_yds', 'rush_td',
        'tackles_solo', 'tackles_combined', 'tackles_loss', 'tackles_assists',
        'fumbles_forced', 'fumbles_rec', 'fumbles_rec_tds', 'fumbles_rec_yds',
        'sacks', 'def_int', 'def_int_td', 'def_int_yards', 'pass_defended',
        'punt_ret', 'punt_ret_td', 'punt_ret_yds',
        'kick_ret', 'kick_ret_td', 'kick_ret_yds'
    ]
    
    # Combine position dummies with numeric features
    feature_matrix = pd.concat([pos_dummies, data[numeric_features]], axis=1)
    return feature_matrix

def tune_xgboost(X_train, y_train, X_test, y_test):
    best_auc = 0
    best_params = {}
    
    # Grid search parameters matching R code
    for depth in [3, 4, 5, 6]:
        for rounds in [50, 100, 150, 200, 250]:
            model = xgb.XGBClassifier(
                max_depth=depth,
                n_estimators=rounds,
                objective='binary:logistic'
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            
            if auc > best_auc:
                best_auc = auc
                best_params = {'max_depth': depth, 'n_estimators': rounds}
    
    return best_params

def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate model performance using various metrics
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities from model
        threshold: Probability threshold for converting to binary prediction (default 0.5)
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    print("\nConfusion Matrix:")
    print("                 Predicted Not First Round  Predicted First Round")
    print(f"Actually Not FR:         {conf_matrix[0][0]}                    {conf_matrix[0][1]}")
    print(f"Actually FR:             {conf_matrix[1][0]}                    {conf_matrix[1][1]}")
    
    return accuracy, precision, recall, f1, conf_matrix

def xgboost():
    # Get data and masks
    data, train_set, test_set, holdout_set = prepare_data()
    train_data = data[train_set]
    test_data = data[test_set]
    holdout_data = data[holdout_set]
    
    # Create labels (1 if picked in first round, 0 otherwise)
    train_data['label'] = train_data['pick'].apply(lambda x: 1 if x <= 32 else 0)
    test_data['label'] = test_data['pick'].apply(lambda x: 1 if x <= 32 else 0)
    holdout_data['label'] = holdout_data['pick'].apply(lambda x: 1 if x <= 32 else 0)
    
    # Create feature matrices
    X_train = create_feature_matrix(train_data)
    X_test = create_feature_matrix(test_data)
    X_holdout = create_feature_matrix(holdout_data)
    
    # Tune XGBoost model
    best_params = tune_xgboost(
        X_train, train_data['label'],
        X_test, test_data['label']
    )
    
    # Train final model on combined train+test data
    X_combined = pd.concat([X_train, X_test])
    y_combined = pd.concat([train_data['label'], test_data['label']])
    
    final_model = xgb.XGBClassifier(
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        objective='binary:logistic'
    )
    final_model.fit(X_combined, y_combined)
    
    # Make predictions
    data['fr.hat'] = final_model.predict_proba(create_feature_matrix(data))[:, 1]
    
    # Evaluate model on holdout set
    print("\nHoldout Set Performance:")
    evaluate_model(holdout_data['label'], 
                  final_model.predict_proba(X_holdout)[:, 1])
    
    # Print results
    print("\nBest XGBoost Parameters:", best_params)
    print("\nPrediction Results:")
    print(data[['player', 'pos', 'year', 'pick', 'fr.hat']].sort_values('fr.hat', ascending=False).head(20))
    
    # Save results to CSV
    data.to_csv('predictions.csv', index=False)
    print("\nFull results saved to 'predictions.csv'")
    
    return final_model

if __name__ == "__main__":
    model = xgboost()
    # Load and test on 2023 data
    data_2023 = pd.read_feather('data/processed/by_year/2020.0/training.feather')
    
    # Create feature matrix for 2023 data
    X_2023 = create_feature_matrix(data_2023)
    
    # Make predictions
    data_2023['fr.hat'] = model.predict_proba(X_2023)[:, 1]
    
    print("\n2023 Draft Class Predictions:")
    print(data_2023[['player', 'pos', 'pick', 'fr.hat']]
          .sort_values('fr.hat', ascending=False)
          .head(32))
    
    # Save 2023 predictions
    data_2023.to_csv('predictions_2020.csv', index=False)
    print("\n2020 predictions saved to 'predictions_2020.csv'")
    

