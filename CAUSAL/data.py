from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

class AbstractDataset(Dataset):
    def __init__(self, name, X: pd.DataFrame, y: pd.DataFrame):
        self.name = name
        self.X = X
        self.y = y
        self.X_preprocessed = None
        self.y_preprocessed = None
        self.X_Z = None
    
    def download_csv(self):
        # use X, y to download the csv file
        # write the code for this below
        df = pd.concat([self.X, self.y], axis=1)
        df.to_csv(f"./CAUSAL/datasets/{self.name}.csv", index=False)

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
    
    def add_Z(self, y_pred):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        y_prime = self.y_preprocessed.ravel()
        Z = y_prime - y_pred
        self.X_Z = np.hstack((self.X_preprocessed, Z.reshape(-1, 1)))
        return Z
    
    def intervention(self):
        raise NotImplementedError("Subclasses must implement intervention() method.")
    
    def check_correlation(self, significance_threshold=0.05):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X = self.X_Z
        y = self.y_preprocessed

        # Create a DataFrame with features and target
        df = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1]-1)] + ["Z"])
        df["T"] = y
        feature_names = [col for col in df.columns if col not in ['T', 'Z']]
        
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
                    f"Z is correlated with {feature} (Pearson p: {pearson_p:.3f}, Spearman p: {spearman_p:.3f})."
                )
        
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
        
        return all_checks_passed
    
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


