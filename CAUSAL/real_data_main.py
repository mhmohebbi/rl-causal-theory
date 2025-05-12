from datasets import DATASETS
from data import AbstractDataset
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import os

def seeding(seed=5):
    # Seed all random number generators for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_datasets():
    # X, y = get_large()
    # datasets = []

    # for i in range(500, 10500, 500):
    #     X_i = X[:i]
    #     y_i = y[:i]
    #     datasets.append(AbstractDataset(name=f"{i}", X=X_i, y=y_i))
    # return datasets
    #datasets = [WindDataset(), PollenDataset(), DailyDemandForecastingOrdersDataset(), ConcreteCompressiveStrengthDataset(), AirfoilDataset(), AuctionVerificationDataset(), RealEstateDataset(), ParkinsonsTelemonitoringDataset(), FriDataset(), WineQualityDataset()] # AbaloneDataset() 
    
    datasets = DATASETS
    datasets = [dataset for dataset in datasets if dataset.name == "623_fri_c4_1000_10"]
    # Limit each dataset to 1000 samples and preprocess
    i = 1
    for dataset in datasets:
        print(f"Dataset {i}: {dataset.name}")
        i +=1
        # Limit to 1000 samples
        if len(dataset.X) > 1000:
            dataset.X = dataset.X.iloc[:1000]
            dataset.y = dataset.y.iloc[:1000]

        # Preprocess the dataset
        # dataset.preprocess() # not needed (ran later)
    # exit()

    return datasets

def load_baselines():
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        tol=1e-4,
        n_iter_no_change=50,
        alpha=1e-4, 
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
    )

    xgb_model = XGBRegressor(
        n_estimators=1500,      
        max_depth=9,           
        learning_rate=0.01,   
        eval_metric='rmse',
        verbosity=1,   
    )

    linear_reg = LinearRegression(fit_intercept=True)
    ridge_reg = Ridge(alpha=0.1)

    return [xgb_model, mlp, linear_reg, ridge_reg]

def baseline_test(model, X_train, X_test, y_train, y_test):#(dataset: AbstractDataset, aug=None):

    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Test MSE: {:.6f}".format(rmse))
    return model, rmse, y_test, y_pred

def baseline_finetune(base_model, X_train_new, X_test, y_train_new, y_test):
    sample_weights = np.full(len(X_train_new), 0.1)

    base_model.set_params(n_estimators=1500)
    base_model.fit(X_train_new, y_train_new, sample_weight=sample_weights, xgb_model=base_model.get_booster())

    # Predict on the test set
    y_pred = base_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Fine-Tuned Data Augmented Test MSE: {:.6f}".format(rmse))
    return base_model, rmse, y_test, y_pred


def augment_data(baseline_model, dataset: AbstractDataset, X_train, y_train, Z_train, feature_a):
    X_intervention = dataset.intervention(X_train, feature_a)

    X = torch.tensor(X_train, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    Z = torch.tensor(Z_train, dtype=torch.float64)

    y_pred = baseline_model.predict(X_intervention)
    print(f"y_pred shape before reshape: {y_pred.shape}")
    y_pred = y_pred.reshape(-1, 1)  # Reshape to match Z shape
    print(f"y_pred shape after reshape: {y_pred.shape}")
    print(f"Z shape: {Z.shape}")
    y_pred = torch.tensor(y_pred, dtype=torch.float64)

    # Ensure Z has the same shape as y_pred
    if Z.dim() == 3:
        Z = Z.squeeze(1)  # Remove the extra dimension if present

    assert y_pred.shape == Z.shape, f"Shapes of y_pred ({y_pred.shape}) and Z ({Z.shape}) do not match."

    y_counterfactual = y_pred + Z

    X_intervention = X_intervention.detach().numpy()
    y_counterfactual = y_counterfactual.detach().numpy()

    return X_intervention, y_counterfactual

def main():
    seeding()
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")

    datasets = load_datasets()
    for dataset in datasets:
        # dataset.download_csv()

        print(f"Dataset: {dataset.name}")
        print(f"X shape: {dataset.X.shape}")
        print(f"y shape: {dataset.y.shape}")
        print()
        X, y = dataset.preprocess()
        X_train, X_test, y_train, y_test = dataset.split()

        #####
        # print(X_train.describe())
        #####

        baselines = load_baselines()
        baseline_model_names = []
        for baseline_model in baselines:
            print(f"Baseline Model: {baseline_model.__class__.__name__}")
            print()
            baseline_model_name = baseline_model.__class__.__name__
            baseline_model_names.append(baseline_model_name)

            baseline_model, rmse1, y_test1, y_pred1 = baseline_test(baseline_model, X_train, X_test, y_train, y_test)
            y_pred = baseline_model.predict(X) # does not matter for now but should only be done on X_train
            print(f"okkk so {baseline_model_name} is {y_pred.shape}")

            if baseline_model_name in ("Ridge", "LinearRegression") : y_pred = y_pred.reshape(-1)

            Z = dataset.add_Z(y_pred)
            
            print()
            res, feature_a = dataset.check_correlation(baseline_model_name, timestamp)
            print()
            feature_a = 4
            # assert res, "Correlation check failed."

            y_pred_train = baseline_model.predict(X_train)
            y_pred_train = y_pred_train.reshape(-1, 1)  # Reshape to match y_train shape
            Z_train = dataset.add_Z(y_pred_train, y_train)
            y_pred_test = baseline_model.predict(X_test)
            y_pred_test = y_pred_test.reshape(-1, 1)  # Reshape to match y_test shape
            Z_test = dataset.add_Z(y_pred_test, y_test)
            
            
            X_aug_train, y_aug_train = augment_data(baseline_model, dataset, X_train, y_train, Z_train, feature_a)

            X_aug = np.vstack((X_train, X_aug_train))
            y_aug = np.vstack((y_train, y_aug_train))

            # shuffle both X_aug and y_aug
            indices = np.arange(len(X_aug))
            np.random.shuffle(indices)
            X_aug = X_aug[indices]
            y_aug = y_aug[indices]

            _, rmse2, y_test2, y_pred2 = baseline_test(baseline_model, X_aug, X_test, y_aug, y_test)
        
            # _, rmse2, y_test2, y_pred2 =  baseline_finetune(baseline_model, X_aug_train, X_test, y_aug_train, y_test)
            os.makedirs(f'./CAUSAL/results/{timestamp}/{baseline_model_name}', exist_ok=True)

            with open(f'./CAUSAL/results/{timestamp}/{baseline_model_name}/mse_results.csv', 'a') as f:
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
            os.makedirs(f'./CAUSAL/results/{timestamp}/{baseline_model_name}/actual_vs_predicted', exist_ok=True)
            plt.savefig(f'./CAUSAL/results/{timestamp}/{baseline_model_name}/actual_vs_predicted/baseline_actual_vs_predicted_{dataset.name}.png')    
            plt.close()

        plot_rmse_delta(baseline_model_names, timestamp)
        # plot_sample_size_vs_rmse(baseline_model_names)

def plot_rmse_delta(baseline_model_names, timestamp):
    """
    Create a bar chart showing the improvement in RMSE (delta between original and augmented data)
    for each dataset and model.
    """    
    # Dictionary to store the results for each model
    all_data = {}
    
    # Read the CSV files for each model
    for model_name in baseline_model_names:
        file_path = f'./CAUSAL/results/{timestamp}/{model_name}/mse_results.csv'
        try:
            # Using pandas to read the CSV
            df = pd.read_csv(file_path, header=None, names=['dataset', 'mse_original', 'mse_data_aug'])
            
            # Calculate the delta (improvement) - positive means improvement
            df['delta'] = df['mse_original'] - df['mse_data_aug']
            df['percent_improvement'] = (df['delta'] / df['mse_original']) * 100
            
            all_data[model_name] = df
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found.")
    
    if not all_data:
        print("No data found to plot.")
        return
    

    # Get a list of all unique datasets
    all_datasets = []
    for df in all_data.values():
        all_datasets.extend(df['dataset'].tolist())
    unique_datasets = sorted(set(all_datasets))

    # Prepare data for plotting
    x = np.arange(len(unique_datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars for each model
    for i, (model_name, df) in enumerate(all_data.items()):
        # Create a dictionary mapping dataset names to their delta values
        delta_dict = dict(zip(df['dataset'], df['delta']))
        percent_dict = dict(zip(df['dataset'], df['percent_improvement']))
        
        # Get the delta values for all datasets
        deltas = [delta_dict.get(dataset, 0) for dataset in unique_datasets]
        percents = [percent_dict.get(dataset, 0) for dataset in unique_datasets]
        
        # Plot the bars
        offset = width * (i - 0.5 * (len(all_data) - 1))
        rects = ax.bar(x + offset, deltas, width, label=model_name, alpha=0.7)
        
        avg_percent_improvement = np.mean(percents)
        print(f"Model: {model_name}, Avg. Percent Improvement: {avg_percent_improvement:.2f}%")

        # Add percentage labels
        for j, rect in enumerate(rects):
            height = rect.get_height()
            if abs(height) > 0.0000001:  # Only add text if bar has non-zero height
                percent = percents[j]
                ax.annotate(f'{percent:.1f}%',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3 if height > 0 else -10),
                           textcoords="offset points",
                           ha='center', va='bottom' if height > 0 else 'top',
                           rotation=90, fontsize=8)
    
    # Add zero line, labels, title, etc.
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('RMSE Improvement (Original - Augmented)')
    ax.set_title('Improvement in RMSE When Using Augmented Data')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_datasets, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(f'./CAUSAL/results/{timestamp}', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'./CAUSAL/results/{timestamp}/rmse_delta_comparison.png')
    plt.close()
    
    print("RMSE delta comparison plot has been saved.")

if __name__ == "__main__":
    main()
