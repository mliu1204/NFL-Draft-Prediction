import pandas as pd
from prepare_data import prepare_data
# import xgboost as xgb
from sklearn.metrics import roc_auc_score
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

if __name__ == "__main__":
    example()

