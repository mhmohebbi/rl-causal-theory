from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class AbstractDataset(Dataset):
    def __init__(self, name, X: pd.DataFrame, y: pd.DataFrame):
        self.df = pd.concat([X, y], axis=1)
        self.name = name
        self.X = X
        self.y = y
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
        raise NotImplementedError("Subclasses must implement plot_size() method.")
    
    def preprocess(self):
        raise NotImplementedError("Subclasses must implement preprocess() method.")
    
    def split(self, test_size=0.2, X=None, y=None):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        if X is None and y is None:
            X = self.X_preprocessed
            y = self.y_preprocessed

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def add_Z(self, y_pred, y_prime=None):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        if y_prime is None:
            y_prime = self.y_preprocessed
            Z = y_prime.ravel() - y_pred
            self.X_Z = np.hstack((self.X_preprocessed, Z.reshape(-1, 1)))
        else:
            Z = y_prime.ravel() - y_pred

        return Z
    
    def intervention(self):
        raise NotImplementedError("Subclasses must implement intervention() method.")
    
    def check_correlation(self, significance_threshold=0.1):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X = self.X_Z
        y = self.y_preprocessed

        # Create a DataFrame with features and target
        df = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1]-1)] + ["Z"])
        df["T"] = y
        feature_names = [col for col in df.columns if col not in ['T', 'Z']]

        df_copy = df.copy()
        independent_features = self.find_independent_features(df_copy)
        print("Independent features:", independent_features)
        print()

        all_checks_passed = True
        reasons = []
        
        # Check correlations between Z and each feature
        for feature in feature_names:
            pearson_corr, pearson_p = pearsonr(df['Z'], df[feature])
            spearman_corr, spearman_p = spearmanr(df['Z'], df[feature])
            print(f"Feature: {feature}, Pearson: {pearson_corr:.3f} (p={pearson_p:.3f}), Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
            
            # If either test suggests significance, flag it
            if pearson_p < significance_threshold or spearman_p < significance_threshold:
                all_checks_passed = False
                reasons.append(
                    f"Z is correlated with {feature} (Pearson p: {pearson_p:.7f}, Spearman p: {spearman_p:.7f})."
                )
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
        
        # Print overall result
        if all_checks_passed:
            print("All checks passed: Z is not correlated with any feature and is correlated with T.")
        else:
            print("Check failed:", " ".join(reasons))

        model.params = model.params.drop("const")
        model.params = model.params.drop("Z")
        sorted_params = model.params.sort_values(ascending=False)

        feature_to_use = None
        for feature in sorted_params.index:
            if feature not in independent_features:
                feature_to_use = feature
                break
        if feature_to_use is None:
            feature_to_use = sorted_params.index[0]
            
        print(f"Feature to use: {feature_to_use}")
        feature_to_use_index = int(feature_to_use[1:])
        return all_checks_passed, feature_to_use_index
    
        # df = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1]-1)] + ["Z"])
        # df["T"] = y
        # feature_names = [col for col in df.columns if col not in ['T', 'Z']]
        # correlation_results = {}
        # for feature in feature_names:
        #     pearson_corr, pearson_p = pearsonr(df['Z'], df[feature])
        #     spearman_corr, spearman_p = spearmanr(df['Z'], df[feature])
        #     correlation_results[feature] = {
        #         'pearson_corr': pearson_corr,
        #         'pearson_p': pearson_p,
        #         'spearman_corr': spearman_corr,
        #         'spearman_p': spearman_p
        #     }
        #     print(f"Feature: {feature}, Pearson: {pearson_corr:.3f} (p={pearson_p:.3f}), Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")

        # # Augmented Regression for the target T:
        # X_aug = df[feature_names + ['Z']]
        # X_aug = sm.add_constant(X_aug)
        # model = sm.OLS(df['T'], X_aug).fit()
        # print(model.summary())


    def find_independent_features(self, df, corr_threshold=0.5):
        # Compute the absolute correlation matrix
        corr_matrix = df.corr().abs()

        # Optionally, visualize the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title(f"Correlation matrix for {self.name}")
        plt.savefig(f'./CAUSAL/correlations/correlation_matrix_{self.name}.png')

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
