from data import AbstractDataset
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnflow import *
import random
import os

def seeding(seed=42):
    # Seed all random number generators for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ground_truth_Z = None

def linear_function(x):
    return 2 * x + 1.5

def add_noise(y, z):
    return y + z

def load_datasets():
    global ground_truth_Z

    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    datasets = []

    for sample_size in sizes:
        X = np.random.rand(sample_size, 1)
        Z = np.random.beta(5, 5, size=(sample_size, 1))

        # ground_truth_Z = Z

        y = linear_function(X)
        y = add_noise(y, Z)
    # y = np.clip(y, 0, 1)  # Simple clipping

        X = pd.DataFrame(X, columns=['X'])
        y = pd.DataFrame(y, columns=['y'])
        dataset = AbstractDataset(name=f"Linear-{sample_size}", X=X, y=y)
        datasets.append(dataset)

    return datasets 


def load_baselines():
    mlp = MLPRegressor(
        hidden_layer_sizes=(16, 8),
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

    class LinearModel:
        def __init__(self):
            pass

        def fit(self, X, y):
            print("Nothing to train")
            pass

        def predict(self, X):
            return linear_function(X)
        
    # return [LinearModel()]

    return [xgb_model, mlp]

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


def augment_data(baseline_model, dataset: AbstractDataset, X_train, Z_train, feature_a):
    X_intervention = dataset.intervention(X_train, feature_a)

    Z = torch.tensor(Z_train, dtype=torch.float64).unsqueeze(1)

    y_pred = linear_function(X_intervention)
    y_pred = torch.tensor(y_pred, dtype=torch.float64).unsqueeze(1)

    print(Z.shape, y_pred.shape)
    assert y_pred.shape == Z.shape, "Shapes of y_pred and Z do not match."

    y_counterfactual = add_noise(y_pred, Z).squeeze(-1)

    X_intervention = X_intervention.detach().numpy()
    y_counterfactual = y_counterfactual.detach().numpy()

    return X_intervention, y_counterfactual

def main():
    global ground_truth_Z

    seeding()

    datasets = load_datasets()
    for dataset in datasets:
        # dataset.download_csv()

        print(f"Dataset: {dataset.name}")
        print(f"X shape: {dataset.X.shape}")
        print(f"y shape: {dataset.y.shape}")
        print()
        X, y = dataset.preprocess()
        X_train, X_test, y_train, y_test = dataset.split()

        baselines = load_baselines()
        baseline_model_names = []
        for baseline_model in baselines:
            print(f"Baseline Model: {baseline_model.__class__.__name__}")
            print()
            baseline_model_name = "simulated/" + baseline_model.__class__.__name__
            baseline_model_names.append(baseline_model_name)

            baseline_model, rmse1, y_test1, y_pred1 = baseline_test(baseline_model, X_train, X_test, y_train, y_test)

            y_pred = linear_function(X)
            y_pred = y_pred.reshape(1, -1)
            Z = dataset.add_Z(y_pred)
            Z = Z.reshape(-1, 1)

            # assert np.allclose(Z, ground_truth_Z), "Z is not the same as ground_truth_Z"

            print()
            #res, feature_a = dataset.check_correlation(baseline_model_name)
            print()
            # assert res, "Correlation check failed."
            # assert feature_a is not None, "Feature to change is None."
            feature_a = 0
            y_pred_train = linear_function(X_train)
            y_pred_train = y_pred_train.reshape(-1, 1) # remove later
            Z_train = dataset.add_Z(y_pred_train, y_train) 

            # y_pred_test = baseline_model.predict(X_test)

            # Z_test = dataset.add_Z(y_pred_test, y_test)
            
            X_aug_train, y_aug_train = augment_data(baseline_model, dataset, X_train, Z_train, feature_a)

            X_aug = np.vstack((X_train, X_aug_train))
            y_aug = np.vstack((y_train, y_aug_train))

            # shuffle both X_aug and y_aug
            indices = np.arange(len(X_aug))
            np.random.shuffle(indices)
            X_aug = X_aug[indices]
            y_aug = y_aug[indices]

            _, rmse2, y_test2, y_pred2 = baseline_test(baseline_model, X_aug, X_test, y_aug, y_test)
        
            # _, rmse2, y_test2, y_pred2 =  baseline_finetune(baseline_model, X_aug_train, X_test, y_aug_train, y_test)
            os.makedirs(f'./CAUSAL/results/{baseline_model_name}/sample_sizes', exist_ok=True)

            with open(f'./CAUSAL/results/{baseline_model_name}/sample_sizes/mse_results.csv', 'a') as f:
                f.write(f"{dataset.name},{rmse1},{rmse2}\n")

            # create a plot
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test1, y_pred1, alpha=0.1, label='Using Original Data', color='blue')
            plt.scatter(y_test2, y_pred2, alpha=0.1, label='Using Augmented Data', color='red')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Actual vs. Predicted on Test Sets of {dataset.name}')
            plt.plot([0, 6], [0, 6], 'r--')
            plt.legend()
            os.makedirs(f'./CAUSAL/results/{baseline_model_name}/sample_sizes/actual_vs_predicted', exist_ok=True)
            plt.savefig(f'./CAUSAL/results/{baseline_model_name}/sample_sizes/actual_vs_predicted/baseline_actual_vs_predicted_{dataset.name}.png')    
            plt.close()

        plot_rmse_delta(baseline_model_names)
        plot_sample_size_vs_rmse(baseline_model_names)



def plot_sample_size_vs_rmse(baseline_model_names):
    """
    Create a scatter plot showing the relationship between sample size and RMSE delta for each dataset and model.
    """
    # Dictionary to store the results for each model
    all_data = {}

    # Read the CSV files for each model
    for model_name in baseline_model_names:
        file_path = f'./CAUSAL/results/{model_name}/sample_sizes/mse_results.csv'
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

    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot scatter points for each model
    for i, (model_name, df) in enumerate(all_data.items()):
        # Plot the scatter points
        ax.scatter(df['dataset'], df['percent_improvement'], label=model_name, alpha=0.7)

        # Plot a smooth trend line
        sorted_df = df.sort_values(by='dataset')
        ax.plot(sorted_df['dataset'], sorted_df['percent_improvement'], linestyle='-', marker='o', alpha=0.7)

    # Add zero line, labels, title, etc.
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('RMSE Improvement (%)')
    ax.set_title('Sample Size vs. RMSE Improvement')
    ax.legend()

    fig.tight_layout()

    plt.savefig('./CAUSAL/results/sample_size_vs_rmse.png')
    plt.close()

    print("Sample size vs. RMSE plot has been saved.")

def plot_rmse_delta(baseline_model_names):
    """
    Create a bar chart showing the improvement in RMSE (delta between original and augmented data)
    for each dataset and model.
    """    
    # Dictionary to store the results for each model
    all_data = {}
    
    # Read the CSV files for each model
    for model_name in baseline_model_names:
        file_path = f'./CAUSAL/results/{model_name}/sample_sizes/mse_results.csv'
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
    os.makedirs('./CAUSAL/results', exist_ok=True)
    
    # Save the plot
    plt.savefig('./CAUSAL/results/rmse_delta_comparison.png')
    plt.close()
    
    print("RMSE delta comparison plot has been saved.")

if __name__ == "__main__":
    main()
