import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class MixtureOfExpertsTrainer:
    def __init__(self, data, batch_size=32, num_epochs=20):
        self.data = data
        self.data_len = len(data)
        self.model = None
        self.C2_features = None
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

        # Input for gating network (features from C2)
        self.C2_features = [col for col in X.columns if any(
            feature in col for feature in ['Age_', 'Gender_', 'Location_', 'PurchaseHistory_', 'DeviceType_'])]

        # Extract C2 features for gating network
        X_train_C2 = X_train[self.C2_features].values.astype(np.float32)
        X_test_C2 = X_test[self.C2_features].values.astype(np.float32)

        # Extract all features for expert networks
        X_train_all = X_train.values.astype(np.float32)
        X_test_all = X_test.values.astype(np.float32)

        # Convert numpy arrays to torch tensors
        X_train_C2 = torch.tensor(X_train_C2, dtype=torch.float32)
        X_test_C2 = torch.tensor(X_test_C2, dtype=torch.float32)

        X_train_all = torch.tensor(X_train_all, dtype=torch.float32)
        X_test_all = torch.tensor(X_test_all, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_C2, X_train_all, y_train)
        test_dataset = TensorDataset(X_test_C2, X_test_all, y_test)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Save test targets for later evaluation
        self.y_test_numpy = y_test.squeeze().numpy()

    def train_model(self):
        # Number of latent classes U
        num_U = 4  # U ∈ {1,2,3,4}

        # Input sizes
        input_size_C2 = self.train_loader.dataset.tensors[0].shape[1]
        input_size_all = self.train_loader.dataset.tensors[1].shape[1]

        class MixtureOfExperts(nn.Module):
            def __init__(self, input_size_all, input_size_C2, num_U):
                super(MixtureOfExperts, self).__init__()
                self.num_U = num_U
                # Gating network
                self.gating_hidden = nn.Linear(input_size_C2, 64)
                self.gating_output = nn.Linear(64, num_U)

                # Expert networks
                self.experts_hidden = nn.ModuleList([nn.Linear(input_size_all, 64) for _ in range(num_U)])
                self.experts_output = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_U)])

            def forward(self, input_C2, input_all):
                # Gating network
                gating_hidden = F.relu(self.gating_hidden(input_C2))
                gating_probs = F.softmax(self.gating_output(gating_hidden), dim=1)  # Shape: (batch_size, num_U)

                # Expert networks
                expert_outputs = []
                for i in range(self.num_U):
                    h = F.relu(self.experts_hidden[i](input_all))
                    out = torch.sigmoid(self.experts_output[i](h))
                    out = self.experts_output[i](h)
                    expert_outputs.append(out)

                # Concatenate expert outputs along dimension 1
                expert_outputs = torch.cat(expert_outputs, dim=1)  # Shape: (batch_size, num_U)

                # Multiply gating_probs and expert_outputs element-wise
                weighted_expert_outputs = gating_probs * expert_outputs  # Shape: (batch_size, num_U)

                # Sum over experts (dimension 1)
                final_output = torch.sum(weighted_expert_outputs, dim=1, keepdim=True)  # Shape: (batch_size, 1)

                return final_output

        self.model = MixtureOfExperts(input_size_all=input_size_all, input_size_C2=input_size_C2, num_U=num_U)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.train_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for batch_C2, batch_all, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_C2, batch_all)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * batch_C2.size(0)
            epoch_train_loss /= len(self.train_loader.dataset)
            self.train_losses.append(epoch_train_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {epoch_train_loss:.4f}")

    def evaluate_model(self):
        # Predict on test data
        self.model.eval()
        y_pred_test = []
        with torch.no_grad():
            for batch_C2, batch_all, _ in self.test_loader:
                outputs = self.model(batch_C2, batch_all)
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
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(f'./marginalizing/loss_during_training_{self.data_len}.png')
        plt.close()

        # Plot actual vs predicted
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test_numpy, self.y_pred_test, alpha=0.5)
        plt.xlabel('Actual R')
        plt.ylabel('Predicted R')
        plt.title('Actual vs. Predicted R on Test Data')
        plt.plot([0,1],[0,1], 'r--') 
        plt.savefig(f'./marginalizing/actual_vs_predicted_{self.data_len}.png')
        plt.close()

        # Residuals
        residuals = self.y_test_numpy - self.y_pred_test
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution on Test Set')
        plt.savefig(f'./marginalizing/residuals_distribution_{self.data_len}.png')
        plt.close()

    def plotting_data(self):
        return self.y_test_numpy, self.y_pred_test
