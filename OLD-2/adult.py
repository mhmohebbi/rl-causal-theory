import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 

class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocess_data():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # If missing values are marked as " ?", replace them with np.nan and drop such rows
    X.replace(" ?", np.nan, inplace=True)
    rows_to_drop = X[X.isnull().any(axis=1)].index

    # Drop the rows from X and y
    X_dropped = X.drop(rows_to_drop)
    y_dropped = y.drop(rows_to_drop)

    X = X.drop(columns=['fnlwgt'])
    X = X.drop(columns=['education-num'])


    # Define which columns are categorical vs. numerical
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    numerical_features = [
        'age', 'capital-gain', 'capital-loss', 'hours-per-week'
    ]

    # Preprocess numerical features using StandardScaler
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[numerical_features])

    # Preprocess categorical features using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(X[categorical_features])

    # Combine the processed numerical and categorical features
    X_processed = np.hstack((X_num, X_cat))

    # Encode the target variable: assuming income labels are '>50K' and '<=50K'
    # Adjust the encoding if the labels differ in your fetched dataset.
    y_encoded = (y == '>50K').astype(int).values

    print("X shape:", X_processed.shape)
    print("y shape:", y_encoded.shape)
    return X_processed, y_encoded

def split(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def baseline():
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = split(X, y)
    train_dataset = AdultDataset(X_train, y_train)
    test_dataset = AdultDataset(X_test, y_test)

    # Create DataLoaders if needed
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Set up the XGBoost classifier as the baseline model
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.0001,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Train the model without early stopping
    xgb_model.fit(
        X_train, 
        y_train,
        # eval_set=[(X_test, y_test)],
        verbose=True
    )

    # Evaluate the model on the test set
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    baseline()