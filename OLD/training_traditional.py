import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

class MLPRegressorTrainer:
    def __init__(self, data):
        self.data = data
        self.data_len = len(data)
        self.model = None
        self.encoder = OneHotEncoder(sparse_output=False) #drop='first',
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None

    def preprocess_data(self, test_indices=None):
        X = self.data[['A', 'TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender', 'Location',
                       'PurchaseHistory', 'DeviceType']]
        y = self.data['R']

        categorical_features = ['TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender',
                                'Location', 'PurchaseHistory', 'DeviceType']

        X_encoded = self.encoder.fit_transform(X[categorical_features])
        encoded_feature_names = self.encoder.get_feature_names_out(categorical_features)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

        X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
        X_final = pd.concat([X_numeric, X_encoded_df], axis=1)

        if test_indices is not None:
            self.X_train = X_final.loc[~X_final.index.isin(test_indices)]
            self.X_test = X_final.loc[X_final.index.isin(test_indices)]
            self.y_train = y.loc[~y.index.isin(test_indices)]
            self.y_test = y.loc[y.index.isin(test_indices)]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

        # Convert data to numpy arrays
        self.X_train = self.X_train.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)

    def train_model(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
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
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

        mse_train = mean_squared_error(self.y_train, self.y_pred_train)
        r2_train = r2_score(self.y_train, self.y_pred_train)

        mse_test = mean_squared_error(self.y_test, self.y_pred_test)
        r2_test = r2_score(self.y_test, self.y_pred_test)

        print(f'Training MSE: {mse_train:.4f}, R^2: {r2_train:.4f}')
        print(f'Testing MSE: {mse_test:.4f}, R^2: {r2_test:.4f}')

        return mse_test, r2_test

    def plot_results(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.y_pred_test, alpha=0.5)
        plt.xlabel('Actual R')
        plt.ylabel('Predicted R')
        plt.title('Actual vs. Predicted R on Test Set')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.savefig(f'./traditional/actual_vs_predicted_{self.data_len}.png')
        plt.close()

        residuals = self.y_test - self.y_pred_test

        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution on Test Set')
        plt.savefig(f'./traditional/residuals_distribution_{self.data_len}.png')
        plt.close()
        
    def plotting_data(self):
        return self.y_test, self.y_pred_test