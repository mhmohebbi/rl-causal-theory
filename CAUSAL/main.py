from datasets import *
from data import AbstractDataset
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
from cnflow import *

def load_datasets():
    datasets = [AuctionVerificationDataset(), ParkinsonsTelemonitoringDataset(), WineQualityDataset(), AbaloneDataset()]
    return datasets

def baseline_test(dataset: AbstractDataset, aug=None):
    # assumse already preprocessed
    if aug is not None:
        X, y = aug
        X_train, X_test, y_train, y_test = dataset.split(X=X, y=y)
    else:
        X_train, X_test, y_train, y_test = dataset.split()
    
    xgb_model = XGBRegressor(
        n_estimators=1500,      # number of trees built
        max_depth=6,           # maximum depth of each tree
        learning_rate=0.01,   # shrinkage factor (lower values require more trees)
        eval_metric='rmse'     # evaluation metric for regression (Root Mean Squared Error)
    )
    xgb_model.fit(X_train, y_train, verbose=True)

    # Predict on the test set
    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Test MSE: {:.2f}".format(rmse))
    return xgb_model, rmse, y_test, y_pred

def train_flow(dataset: AbstractDataset, residual_noise: np.ndarray):
    t_dim = dataset.y_preprocessed.shape[1] # Dimensionality of target T
    f_dim = dataset.X_preprocessed.shape[1] # Dimensionality of features F
    cond_dim = f_dim     # Now conditioning is only on F
    hidden_dim = 64
    num_coupling_layers = 4
    batch_size = 32
    learning_rate = 1e-3
    num_epochs =75

    X_train, X_test, y_train, y_test = dataset.split()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(y_train, X_train)
    test_dataset = torch.utils.data.TensorDataset(y_test, X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    flow = ConditionalNormalizingFlow(t_dim=t_dim, cond_dim=cond_dim, 
                                    hidden_dim=hidden_dim,
                                    num_coupling_layers=num_coupling_layers)
    optimizer = optim.Adam(flow.parameters(), lr=learning_rate)

    flow.fit(train_loader, optimizer, num_epochs=num_epochs, device='cpu')
    flow.evaluate(test_loader, device='cpu')
    mse, T_recon, z_learned = flow.evaluate_recon(y_test, X_test)

    # compare the learned latent z with the reference residual noise Z
    Z = torch.tensor(residual_noise, dtype=torch.float32).unsqueeze(1)
    flow.plot_results(y_test, T_recon, z_learned, dataset.name, z_reference=Z)

    return flow

def augment_data(flow: ConditionalNormalizingFlow, dataset: AbstractDataset):
    X_intervention, mapping = dataset.intervention()
    X = torch.tensor(dataset.X_preprocessed, dtype=torch.float32)
    y = torch.tensor(dataset.y_preprocessed, dtype=torch.float32)

    z_original, _ = flow.forward(y, X)
    z_intervention_mapped = z_original[mapping]

    T_intervention, _ = flow.inverse(z_intervention_mapped, X_intervention)

    X_intervention = X_intervention.detach().numpy()
    T_intervention = T_intervention.detach().numpy()

    return X_intervention, T_intervention

def main():
    datasets = load_datasets()
    for dataset in datasets:
        print(f"Dataset: {dataset.name}")
        print(f"X shape: {dataset.X.shape}")
        print(f"y shape: {dataset.y.shape}")
        print()
        X, y = dataset.preprocess()

        baseline_model, rmse1, y_test1, y_pred1 = baseline_test(dataset)
        y_pred = baseline_model.predict(X)
        Z = dataset.add_Z(y_pred)
        
        print()
        res = dataset.check_correlation()
        print()
        assert res, "Correlation check failed."
        # exit()
        
        flow = train_flow(dataset, Z)
        print()
        X_aug, y_aug = augment_data(flow, dataset)
        _, rmse2, y_test2, y_pred2 = baseline_test(dataset, aug=(X_aug, y_aug))

        with open('./CAUSAL/results/mse_results.csv', 'a') as f:
            f.write(f"{dataset.name},{rmse1},{rmse2}\n")

        # create a plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test1, y_pred1, alpha=0.1, label='Using Original Data', color='blue')
        plt.scatter(y_test2, y_pred2, alpha=0.1, label='Using Augmented Data', color='red')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs. Predicted on Test Sets of {dataset.name}')
        size = dataset.plot_size()
        plt.plot([0, size], [0, size], 'r--')
        plt.legend()
        plt.savefig(f'./CAUSAL/results/baseline_actual_vs_predicted_{dataset.name}.png')    
        plt.close()

if __name__ == "__main__":
    main()
