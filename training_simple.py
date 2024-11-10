import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class SimpleNNTrainer:
    def __init__(self, data, batch_size=32, num_epochs=20):
        self.data = data
        self.data_len = len(data)
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.train_losses = []
        self.y_test_numpy = None
        self.y_pred_test = None
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.encoder = OneHotEncoder(sparse_output=False)

    def preprocess_data(self):
        X = self.data[['A', 'TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender', 'Location',
                       'PurchaseHistory', 'DeviceType']]
        y = self.data['R'].values

        categorical_features = ['TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender',
                                'Location', 'PurchaseHistory', 'DeviceType']

        # One-hot encode categorical features
        X_encoded = self.encoder.fit_transform(X[categorical_features])
        encoded_feature_names = self.encoder.get_feature_names_out(categorical_features)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

        # Combine numerical and encoded categorical features
        X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
        X = pd.concat([X_numeric, X_encoded_df], axis=1)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Convert dataframes to numpy arrays
        X_train = X_train.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)

        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        # Convert numpy arrays to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # Create DataLoaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size)

        # Save test targets for evaluation
        self.y_test_numpy = y_test.squeeze().numpy()

    def train_model(self):
        # Define the neural network architecture
        input_size = self.train_loader.dataset.tensors[0].shape[1]
        hidden_size = 64
        output_size = 1

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # # Initialize weights to match scikit-learn's MLPRegressor
        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         fan_in = m.weight.size(1)
        #         fan_out = m.weight.size(0)
        #         factor = 6.0
        #         init_bound = np.sqrt(factor / (fan_in + fan_out))
        #         nn.init.uniform_(m.weight, -init_bound, init_bound)
        #         if m.bias is not None:
        #             nn.init.uniform_(m.bias, -init_bound, init_bound)

        # self.model.apply(init_weights)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.train_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0)
            epoch_train_loss /= len(self.train_loader.dataset)
            self.train_losses.append(epoch_train_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {epoch_train_loss:.4f}")

    def evaluate_model(self):
        # Predict on test data
        self.model.eval()
        y_pred_test = []
        with torch.no_grad():
            for batch_X, _ in self.test_loader:
                outputs = self.model(batch_X)
                y_pred_test.append(outputs)
        self.y_pred_test = torch.cat(y_pred_test, dim=0).squeeze().numpy()

        # Calculate MSE and R2 score
        mse_test = mean_squared_error(self.y_test_numpy, self.y_pred_test)
        r2_test = r2_score(self.y_test_numpy, self.y_pred_test)

        print(f"Testing MSE: {mse_test:.4f}, R2: {r2_test:.4f}")

        return mse_test, r2_test

    def plot_results(self):
        # Plot training loss
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(f'./traditional/loss_during_training_{self.data_len}.png')
        plt.close()

        # Plot actual vs. predicted values
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test_numpy, self.y_pred_test, alpha=0.5)
        plt.xlabel('Actual R')
        plt.ylabel('Predicted R')
        plt.title('Actual vs. Predicted R on Test Data')
        min_val = min(self.y_test_numpy.min(), self.y_pred_test.min())
        max_val = max(self.y_test_numpy.max(), self.y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.savefig(f'./traditional/actual_vs_predicted_{self.data_len}.png')
        plt.close()

        # Plot residuals distribution
        residuals = self.y_test_numpy - self.y_pred_test
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution on Test Set')
        plt.savefig(f'./traditional/residuals_distribution_{self.data_len}.png')
        plt.close()

    def plotting_data(self):
        return self.y_test_numpy, self.y_pred_test
