from scipy.stats import pearsonr, spearmanr
import os
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
from causal import CausalGraphLearner
import matplotlib.pyplot as plt

p1 = 0.025
p2 = 0.05
class AbstractDataset(Dataset):
    def __init__(self, name, X: pd.DataFrame, y: pd.DataFrame):
        # Concatenate X and y, then drop any rows with missing values and reset the index
        self.df = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)
        self.df = self.df.drop_duplicates()
        # Update X and y to reflect the cleaned dataframe
        self.X = self.df[X.columns]
        self.y = self.df[y.columns]

        self.name = name

        self.X_preprocessed = None
        self.y_preprocessed = None
        self.X_Z = None

        self.scaler_X = None
        self.scaler_y = None
        self.normalizer_X = None
        self.normalizer_y = None
        # optional i guess
        # print(X.head())
        # print(y.head())

    def inverse_transform_x(self, x):
        x_standardized = self.normalizer_X.inverse_transform(x)
        x_original = self.scaler_X.inverse_transform(x_standardized)
        return x_original
    
    def transform_x(self, x):
        x_standardized = self.scaler_X.transform(x)
        x_normalized = self.normalizer_X.transform(x_standardized)
        return x_normalized
    
    def inverse_transform_y(self, y):
        y_standardized = self.normalizer_y.inverse_transform(y)
        y_original = self.scaler_y.inverse_transform(y_standardized)
        return y_original
    
    def transform_y(self, y):
        y_standardized = self.scaler_y.transform(y)
        y_normalized = self.normalizer_y.transform(y_standardized)
        return y_normalized
    
    def get_range(self, feature_index):
        range = (self.df.iloc[:, feature_index].min(), self.df.iloc[:, feature_index].max())

        print(f"Range of feature {feature_index}: {range}")
        return range
    
    def download_csv(self):
        self.df.to_csv(f"./CAUSAL/datasets/{self.name}.csv", index=False)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def plot_size(self):
        return 1
    
    def preprocess(self):        
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        # self.normalizer_X = MinMaxScaler()
        # self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        # self.scaler_y = StandardScaler()
        # self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y.values)#fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed
    
    def split(self, test_size=0.2, X=None, y=None):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        if X is None and y is None:
            X = self.X_preprocessed
            y = self.y_preprocessed

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test
    
    def add_Z(self, y_pred, y_prime=None):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        if y_prime is None:
            y_prime = self.y_preprocessed
            Z = y_prime.ravel() - y_pred
            self.X_Z = np.hstack((self.X_preprocessed, Z.reshape(-1, 1)))
        else:
            assert y_prime.shape == y_pred.shape, "y_prime and y_pred must have the same shape."
            Z = y_prime - y_pred

        return Z
    
    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        return X_intervention

    def check_casual_graph(self, baseline_model_name, timestamp):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X = self.X_Z
        y = self.y_preprocessed

        # target_name = self.df.columns[-1]
        # feature_names = self.df.columns[:-1]
        # new_col_names = feature_names.tolist() + ['Z']
        # df = pd.DataFrame(X, columns=new_col_names)
        # df[target_name] = y

        df = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1]-1)] + ["Z"])
        df["T"] = y
        learner = CausalGraphLearner(alpha=0.05)
        learner.fit(df)

        graph = learner.get_graph()
        # print()
        # print(graph)
        # print()
        graph_viz = learner.plot_graph()
        graph_viz.render(f'./CAUSAL/results/{timestamp}/causal_graphs/{baseline_model_name}/{self.name}', format='png', cleanup=True)

        connected_to_z = learner.get_adjacent_nodes('Z')
        print("Nodes connected to Z:", connected_to_z)
        not_connected_to_z = learner.get_non_adjacent_nodes('Z')
        print("Nodes NOT connected to Z:", not_connected_to_z)

        print()
        return not_connected_to_z

    def check_correlation(self, baseline_model_name, timestamp, significance_threshold=0.001):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        not_connected_to_z = self.check_casual_graph(baseline_model_name, timestamp)

        X = self.X_Z
        y = self.y_preprocessed

        # Create a DataFrame with features and target
        df = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1]-1)] + ["Z"])
        df["T"] = y
        feature_names = [col for col in df.columns if col not in ['T', 'Z']]

        df_copy = df.copy()
        independent_features = self.find_independent_features(df_copy, baseline_model_name, timestamp)
        # print("Independent features:", independent_features)
        # print()

        all_checks_passed = True
        reasons = []

        abs_spearman_corrs = {}
        # Check correlations between Z and each feature
        for feature in feature_names:
            pearson_corr, pearson_p = pearsonr(df['Z'], df[feature])
            spearman_corr, spearman_p = spearmanr(df['Z'], df[feature])
            print(f"Feature: {feature}, Pearson: {pearson_corr:.3f} (p={pearson_p:.3f}), Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
            
            abs_spearman_corrs[feature] = abs(spearman_corr)
            # If either test suggests significance, flag it
            bad_candidates = []
            if pearson_p < significance_threshold or spearman_p < significance_threshold:
                bad_candidates.append(feature)
                reasons.append(
                    f"Z is correlated with {feature} (Pearson p: {pearson_p:.7f}, Spearman p: {spearman_p:.7f})."
                )
                print(f"Z is correlated with {feature} (Pearson p: {pearson_p:.7f}, Spearman p: {spearman_p:.7f}).")
        print()

        # Regression to check that Z is correlated with the target T
        X_aug = df[feature_names + ['Z']]
        X_aug = sm.add_constant(X_aug)
        model = sm.OLS(df['T'], X_aug).fit()
        print(model.summary())
        
        z_p_value = model.pvalues['Z']
        if z_p_value > significance_threshold:
            all_checks_passed = False
            reasons.append(f"Z is not significantly correlated with target T (p-value: {z_p_value:.3f}).")
            print(f"Z is not significantly correlated with target T (p-value: {z_p_value:.3f}).")

        model.params = model.params.drop("const")
        model.params = model.params.drop("Z")

        abs_params = model.params.abs()
        candidate_features = set(not_connected_to_z).intersection(abs_params.index)
        candidate_features = [feature for feature in candidate_features if feature not in bad_candidates]
        if not candidate_features:
            print("Check failed: no suitable features found that meet all criteria. (all connected to Z)")
            all_checks_passed = False
            return all_checks_passed, None
        
        # Calculate a combined score for each feature
        # Higher score = better feature (higher model coefficient and lower correlation with Z)
        feature_scores = {}
        for feature in candidate_features:
            model_coef = abs_params[feature]
            z_corr = abs_spearman_corrs[feature]
            # Weight model coefficient positively and Z correlation negatively
            # Normalize the values if needed for more balanced weighting
            feature_scores[feature] = model_coef - z_corr

        # Select feature with highest combined score
        feature_to_use = max(feature_scores.items(), key=lambda x: x[1])[0]
        print()
        print(f"Feature to use: {feature_to_use}")
        print(f"Selected feature model coefficient: {abs_params[feature_to_use]:.4f}")
        print(f"Selected feature correlation with Z: {abs_spearman_corrs[feature_to_use]:.4f}")
        

        feature_to_use_index = int(feature_to_use[1:])
        return all_checks_passed, feature_to_use_index

    def find_independent_features(self, df, baseline_model_name, timestamp, corr_threshold=0.5):
        # Compute the absolute correlation matrix
        corr_matrix = df.corr().abs()

        # Optionally, visualize the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title(f"Correlation matrix for {self.name}")
        os.makedirs(f'./CAUSAL/results/{timestamp}/correlations/{baseline_model_name}', exist_ok=True)
        plt.savefig(f'./CAUSAL/results/{timestamp}/correlations/{baseline_model_name}/correlation_matrix_{self.name}.png')

        # Greedy selection: iterate over features and keep those that are not highly correlated
        selected_features = []
        df = df.drop(columns=['T'])
        df = df.drop(columns=['Z'])
        
        for feature in df.columns:
            corr_matrix[feature].drop("T")
            corr_matrix[feature].drop("Z")
            # If the feature is not highly correlated with any already selected feature, keep it
            # if all(corr_matrix.loc[feature, selected_features] < corr_threshold):
            if (corr_matrix[feature].drop(feature) < corr_threshold).all():
                selected_features.append(feature)

        return selected_features
