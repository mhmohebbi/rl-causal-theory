from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from training_flows import NormalizingFlowsTrainer
import torch
from sklearn.neural_network import MLPRegressor
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def get_preprocessed_data():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 

    X_old = adult.data.features
    y_old = adult.data.targets

    # add y_old to X_old
    X_old['income'] = y_old

    #drop "hours-per-week" from X_old and make it y_old
    y_old = X_old['hours-per-week']
    X_old = X_old.drop(columns=['hours-per-week'])

    X = X_old.dropna()
    y = y_old.loc[X.index]
    
    # Optional checks
    assert not X.isnull().values.any(), "There are still missing values in X."
    assert not y.isnull().values.any(), "There are still missing values in y."

    # 1) One-hot encode all categorical columns in X
    X_encoded = X.copy()

    # Loop through columns to encode categorical ones
    for column in X_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X_encoded[column])

    # Convert boolean columns to integers
    X_encoded = X_encoded.astype(int)

    # drop education column
    X_encoded = X_encoded.drop(columns=['education'])
    X_encoded = X_encoded.drop(columns=['fnlwgt'])
    scaler = StandardScaler()
    X_encoded = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

    # 2)  encode y
    # we want to normalize the target variable
    values = y.unique()
    scaled_values = (values - values.min()) / (values.max() - values.min())
    y_encoded = y.replace(dict(zip(values, scaled_values)))

    # Optionally, check that everything is numeric now
    assert all([pd.api.types.is_numeric_dtype(dt) for dt in X_encoded.dtypes]), \
        "Not all feature columns are numeric."
    if isinstance(y_encoded, pd.DataFrame):
        assert all([pd.api.types.is_numeric_dtype(dt) for dt in y_encoded.dtypes]), \
            "Not all target columns are numeric."

    print(X_encoded.head())
    # turn them into tensors
    X_encoded = torch.tensor(X_encoded.values, dtype=torch.float32)
    y_encoded = torch.tensor(y_encoded.values, dtype=torch.float32).unsqueeze(1)
    return X_encoded, y_encoded

def base_train_model(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(150, 75),
        activation='relu',
        solver='adam',
        max_iter=20,
        batch_size=32,
        learning_rate='constant',
        learning_rate_init=0.001,
        tol=0.0,
        n_iter_no_change=50,
        alpha=0.0, 
        verbose=True,
        random_state=10
    )
    model.fit(X_train, y_train)
    return model

def base_evaluate_model(model, X_test, y_test):
    y_pred_test = model.predict(X_test)


    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f'Testing MSE: {mse_test:.4f}, R^2: {r2_test:.4f}')

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.xlabel('Actual T')
    plt.ylabel('Predicted T')
    plt.title('Actual vs. Predicted T on Test Set')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.savefig(f'./actual_vs_predicted_base.png')
    plt.close()
    return mse_test, r2_test

def main():
    X, y = get_preprocessed_data()
    # make sample smaller by taking only 1000 samples
    X = X[:5000]
    y = y[:5000]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    exit()
    # test on base results
    model = base_train_model(X_train, y_train)
    base_evaluate_model(model, X_test, y_test)

    n_flows = NormalizingFlowsTrainer([],num_U=5)

    n_flows.X_train = X_train
    n_flows.X_test = X_test
    n_flows.y_train = y_train
    n_flows.y_test = y_test

    n_flows.X_dim = X.shape[1]
    n_flows.train_model()
    n_flows.evaluate_model()
    n_flows.plot_results()

    X_aug, y_aug = [], []
    # now test counterfactuals for data augmentation
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]

        a_prime1 = x[0].item() - 40.0
        a_prime2 = x[0].item() - 10.0
        y_prime = n_flows.counterfactual_outcome(x, y, a_prime1)
        print("original outcome: ", y)
        print(f"Counterfactual outcome for {x} with action {a_prime1} is {y_prime}")
        exit()

main()