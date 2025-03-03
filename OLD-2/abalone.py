from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from cnf import *
from cnflow import *

class AbaloneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # regression targets
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocess_abalone():
    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features.copy()
    y = abalone.data.targets.copy()

    # print(X[455:500])
    # print(y.head())

    # Define categorical and numerical features
    categorical_features = ["Sex"]
    numerical_features = ["Length", "Diameter", "Height", "Whole_weight",
                          "Shucked_weight", "Viscera_weight", "Shell_weight"]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[numerical_features])
    
    # One-hot encode the categorical feature
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(X[categorical_features])
    
    # Combine numerical and categorical features
    X_processed = np.hstack((X_num, X_cat))
    
    # print(X_processed[455:500])
    # print(y.head())
    return X_processed, y.values

def predictive_Z(X_, y_):
    X = X_.copy()
    y = y_.copy() 
    # Split into train and test sets (and keep the corresponding original rows for later)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = AbaloneDataset(X_train, y_train)
    test_dataset = AbaloneDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    xgb_model = XGBRegressor(
        n_estimators=1500,      # number of trees built
        max_depth=6,           # maximum depth of each tree
        learning_rate=0.01,   # shrinkage factor (lower values require more trees)
        eval_metric='rmse'     # evaluation metric for regression (Root Mean Squared Error)
    )
    
    # Train the regression model (without early stopping here)
    xgb_model.fit(X_train, y_train, verbose=True)
    
    # Predict on the test set
    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # create a plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted on Test Set')
    plt.plot([0, 21], [0, 21], 'r--')
    plt.savefig('./CAUSAL/actual_vs_predicted_abalone.png')    


    print("Test MSE: {:.2f}".format(rmse))

    y_prime = y.ravel()

    y_pred = xgb_model.predict(X)
    Z = y_prime - y_pred
    
    X = np.hstack((X, Z.reshape(-1, 1)))
    return X, y

def check_correlation(X, y):
    df = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1]-1)] + ["Z"])
    df["T"] = y
    feature_names = [col for col in df.columns if col not in ['T', 'Z']]
    correlation_results = {}
    for feature in feature_names:
        pearson_corr, pearson_p = pearsonr(df['Z'], df[feature])
        spearman_corr, spearman_p = spearmanr(df['Z'], df[feature])
        correlation_results[feature] = {
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p
        }
        print(f"Feature: {feature}, Pearson: {pearson_corr:.3f} (p={pearson_p:.3f}), Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")

    # Augmented Regression for the target T:
    X_aug = df[feature_names + ['Z']]
    X_aug = sm.add_constant(X_aug)
    model = sm.OLS(df['T'], X_aug).fit()
    print(model.summary())

def main():
    X_old, y_old = preprocess_abalone()
    X, y = predictive_Z(X_old, y_old)
    check_correlation(X, y)
    X_old = torch.tensor(X_old, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_old, y, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(y_train, X_train)
    test_dataset = torch.utils.data.TensorDataset(y_test, X_test)

    t_dim = 1            # Dimensionality of target T
    f_dim = 10           # Dimensionality of features F
    cond_dim = f_dim     # Now conditioning is only on F
    hidden_dim = 32
    num_coupling_layers = 4
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    flow = ConditionalNormalizingFlow(t_dim=t_dim, cond_dim=cond_dim, 
                                    hidden_dim=hidden_dim,
                                    num_coupling_layers=num_coupling_layers)
    optimizer = optim.Adam(flow.parameters(), lr=learning_rate)

    flow.fit(train_loader, optimizer, num_epochs=num_epochs, device='cpu')
    flow.evaluate(test_loader, device='cpu')


    # model = ConditionalCNF(cond_dim=10, hidden_dim=64)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # train_conditional_cnf(model, optimizer, dataloader, epochs=5)
    # torch.save(model.state_dict(), './CAUSAL/abalone_cnf.pth')

    # load the model from the saved state
    # model = ConditionalCNF(cond_dim=10, hidden_dim=64)
    # model.load_state_dict(torch.load('./CAUSAL/abalone_cnf.pth'))
    

    # evaluate the model

    mse, T_recon, z_learned = flow.evaluate_recon(y_test, X_test)
    flow.plot_results(y_test, T_recon, z_learned, "dataset")

    # compare the learned latent z with the reference residual noise Z
    Z = X[:, -1]
    Z = torch.tensor(Z, dtype=torch.float32).unsqueeze(1)
    flow.plot_results(y_test, T_recon, z_learned, "dataset", z_reference=Z)

    mapping = list(range(X_old.shape[0]))
    X_intervention = X_old.clone()
    print("Dataset Size Pre Intervention: ", X_intervention.shape)
    for i in range(X_intervention.shape[0]):
        if X_intervention[i, -3] == 1 and X_intervention[i, -2] == 0 and X_intervention[i, -1] == 0:
            X_intervention_copy = X_intervention[i].clone()
            X_intervention_copy[-3] = 0
            X_intervention_copy[-2] = 0
            X_intervention_copy[-1] = 1
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)
            X_intervention_copy[-3] = 0
            X_intervention_copy[-2] = 1
            X_intervention_copy[-1] = 0
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)  # Counterfactual from sample i

        elif X_intervention[i, -3] == 0 and X_intervention[i, -2] == 1 and X_intervention[i, -1] == 0:
            X_intervention_copy = X_intervention[i].clone()
            X_intervention_copy[-3] = 1
            X_intervention_copy[-2] = 0
            X_intervention_copy[-1] = 0
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

            X_intervention_copy[-3] = 0
            X_intervention_copy[-2] = 0
            X_intervention_copy[-1] = 1
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

        elif X_intervention[i, -3] == 0 and X_intervention[i, -2] == 0 and X_intervention[i, -1] == 1:
            X_intervention_copy = X_intervention[i].clone()
            X_intervention_copy[-3] = 1
            X_intervention_copy[-2] = 0
            X_intervention_copy[-1] = 0
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

            X_intervention_copy[-3] = 0
            X_intervention_copy[-2] = 1
            X_intervention_copy[-1] = 0
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

    print("Dataset Size Post Intervention: ", X_intervention.shape)

    z_original, _ = flow.forward(y, X_old)
    z_intervention_mapped = z_original[mapping]

    T_intervention, _ = flow.inverse(z_intervention_mapped, X_intervention)

    X_intervention = X_intervention.detach().numpy()
    T_intervention = T_intervention.detach().numpy()

    # split it again
    X_train, X_test, y_train, y_test = train_test_split(X_intervention, T_intervention, test_size=0.2, random_state=42)

    xgb_model = XGBRegressor(
        n_estimators=1500,      # number of trees built
        max_depth=6,           # maximum depth of each tree
        learning_rate=0.01,   # shrinkage factor (lower values require more trees)
        eval_metric='rmse'     # evaluation metric for regression (Root Mean Squared Error)
    )
    
    # Train the regression model (without early stopping here)
    xgb_model.fit(X_train, y_train, verbose=True)
    
    # Predict on the test set
    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Counterfactual MSE: {:.2f}".format(rmse))

if __name__ == "__main__":
    main()


"""
In theory, if your model is correctly specified and all the important predictors of T are included in X,
then the residuals (or noise) Z should be independent of T and of the other predictors

In practice, though, you might want to test for any remaining association or even a potential 
causal influence from Z.
"""

"""
Thanks, now that we have a dataframe of the original dataset (assume we have X, y now) that contains Z added as a column representing the noise.

I want us to train a conditional continuous normalizing flow. For the normalizing flow, we will drop Z and treat it as a hidden variable even tho we have its values.
This nflow is tasked to be trained to learn a mapping between the target and its latent space z. Ideally we want this latent space to sort of represent the same residual noise Z.
Not sure how that is possible though even.

This conditional normalizing flow should be trained and conditioned on every feature in X (exclude Z, we just keep that for reference.
So that the transformations from z to the target and the target to z are influenced by what the features are.
"""