import pandas as pd
from prepare_data import prepare_data

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def makeXY (scaler, data_path, numBins = 2, isTesting = False, features = ["pos", "year", "n_college_picks",
    "short_college", "age", "bench", "broad", "def_int", "def_int_td",
    "def_int_yards", "forty", "fumbles_forced", "fumbles_rec", "fumbles_rec_tds",
    "fumbles_rec_yds", "games", "height", "kick_ret", "kick_ret_td", "kick_ret_yds",
    "pass_att", "pass_cmp", "pass_defended", "pass_int", "pass_td", "pass_yds",
    "punt_ret", "punt_ret_td", "punt_ret_yds", "rec_td", "rec_yards", "receptions",
    "rush_att", "rush_td", "rush_yds", "sacks", "seasons", "shuttle",
    "tackles_assists", "tackles_combined", "tackles_loss", "tackles_solo",
    "threecone", "vertical", "weight"]):
    df = pd.read_csv(data_path)  # Replace with your actual file name
    df = df.drop(columns=["carav"])
    
    # this is to bin all the picks
    Y = df[["pick"]].copy()  # Make a copy to avoid warnings
    # Define bin edges: 8 bins, last one for 257, second-last for leftovers
    if numBins == 8:
        bin_edges = [1, 37, 73, 109, 145, 181, 217, 256, 257]  # (Each bin spans 36, last bin = 257)
        bin_labels = list(range(8))  # Labels: 0 to 7

        # Apply binning
        df["pick_bin"] = pd.cut(df["pick"], bins=bin_edges, labels=bin_labels, include_lowest=True, right=True)

        df["pick_bin"] = df["pick_bin"].fillna(7)  # Assign a valid bin to NaNs

        # Convert bin labels to integers
        Y = df[["pick_bin"]].astype(int)

        # One-hot encode Y for multi-class classification
        Y = to_categorical(Y, num_classes=8)  # Convert to categorical format
    elif numBins == 2:
        Y = df[["pick"]].applymap(lambda x: 0 if x == 257 else 1)




    X = df[features]
    le = LabelEncoder()
    if "short_college" in features:
        X["short_college"] = le.fit_transform(df["short_college"])
    if "pos" in features:
        X["pos"] = le.fit_transform(df["pos"])

    # Display the first few rows to verify
    print(df.head())
    if isTesting:
        X = scaler.transform(X)
    else: 
        X = scaler.fit_transform(X)
    return X, Y

def nn_bin(train_path, test_path):
    scaler = StandardScaler()
    X_train, y_train = makeXY(scaler, train_path,numBins = 8)
    X_test, y_test = makeXY(scaler,test_path,numBins = 8, isTesting=True)
    pd.DataFrame(X_train).to_csv("X_train_data.csv", index=False)
    pd.DataFrame(y_train).to_csv("y_train_data.csv", index=False)
    pd.DataFrame(X_test).to_csv("X_test_data.csv", index=False)
    pd.DataFrame(y_test).to_csv("y_test_data.csv", index=False)

    model = keras.Sequential([
        Input(shape=(X_train.shape[1],)),  # Input layer
        Dense(16, activation="relu"),      # Hidden layer 1
        Dense(8, activation="relu"),       # Hidden layer 2
        Dense(8, activation="softmax")     # Output layer (8 classes, softmax activation)
    ])


    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Display model summary
    model.summary()
    history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), verbose=1)

    # Evaluate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Make Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Predictions:", y_pred)
    # Save the trained model
    model.save("position_classifier.h5")

    # Load the model later
    loaded_model = keras.models.load_model("position_classifier.h5")

    return 

def nn_binary(train_path, test_path):
    scaler = StandardScaler()
    X_train, y_train = makeXY(scaler, train_path,numBins = 2)
    X_test, y_test = makeXY(scaler,test_path,numBins = 2, isTesting= True)
    pd.DataFrame(X_train).to_csv("X_train_data.csv", index=False)
    pd.DataFrame(y_train).to_csv("y_train_data.csv", index=False)
    pd.DataFrame(X_test).to_csv("X_test_data.csv", index=False)
    pd.DataFrame(y_test).to_csv("y_test_data.csv", index=False)

    model = keras.Sequential([
        Input(shape=(X_train.shape[1],)),  # Input layer
        Dense(16, activation="relu"),      # Hidden layer 1
        Dense(8, activation="relu"),       # Hidden layer 2
        Dense(1, activation="sigmoid")     # Output layer 
    ])


    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Display model summary
    model.summary()
    history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), verbose=1)

    # Evaluate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Make Predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("Predictions:", y_pred.flatten())

    # Save the trained model
    model.save("position_classifier.h5")

    # Load the model later
    loaded_model = keras.models.load_model("position_classifier.h5")

    return 


def example():
    # Returns data and masks. Example below is how to separate data with masks.
    train_path = "train_data.csv"
    test_path = "test_data.csv"
    data, train_set, test_set, holdout_set = prepare_data()
    train_data = data[train_set]
    test_data = data[test_set]
    holdout_data = data[holdout_set]

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    holdout_data.to_csv("holdout_data.csv", index=False)

    print(train_data.head())
    print(test_data.head())
    print(holdout_data.head())
    
    return 
    
    # The label is the pick. In the article that we are following, they make the label whether the player
    # was picked in the first round or not.
    # We can do this by using the pick column.
    # Pick 257 means undrafted, as there are 256 picks in the first round.
    # train_data['label'] = train_data['pick'].apply(lambda x: 1 if x <= 32 else 0)
    # test_data['label'] = test_data['pick'].apply(lambda x: 1 if x <= 32 else 0)
    # holdout_data['label'] = holdout_data['pick'].apply(lambda x: 1 if x <= 32 else 0)

if __name__ == "__main__":
    # example()
    train_path = "train_data.csv"
    test_path = "test_data.csv"
    nn_binary(train_path, test_path)
    # nn_bin(train_path, test_path)
