import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class MixtureOfExpertsTrainer:
    def __init__(self, data):
        self.data = data
        self.data_len = len(data)
        self.model = None
        self.C2_features = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_losses = []
        self.val_losses = []
        self.y_test_numpy = None
        self.y_pred_test = None

    def preprocess_data(self):
        # Drop 'U' if present and use 'R' as is
        df = self.data.copy()
        if 'U' in df.columns:
            df = df.drop(columns=['U'])  # Ensure 'U' is not included

        # List of categorical features
        categorical_features = ['TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender',
                                'Location', 'PurchaseHistory', 'DeviceType']

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features)

        # Split into features and target
        X = df_encoded.drop(columns=['R'])
        y = df_encoded['R'].values

        # Split into training and testing sets
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Input for gating network (features from C2)
        self.C2_features = [col for col in X.columns if any(
            feature in col for feature in ['Age_', 'Gender_', 'Location_', 'PurchaseHistory_', 'DeviceType_'])]

        # Extract C2 features for gating network
        X_train_C2 = X_train[self.C2_features].values
        X_val_C2 = X_val[self.C2_features].values
        X_test_C2 = X_test[self.C2_features].values

        # Convert dataframes to numpy arrays
        X_train_all = X_train.values
        X_val_all = X_val.values
        X_test_all = X_test.values

        # Convert numpy arrays to torch tensors
        X_train_C2 = np.array(X_train_C2, dtype=np.float32)
        X_val_C2 = np.array(X_val_C2, dtype=np.float32)
        X_test_C2 = np.array(X_test_C2, dtype=np.float32)

        X_train_all = np.array(X_train_all, dtype=np.float32)
        X_val_all = np.array(X_val_all, dtype=np.float32)
        X_test_all = np.array(X_test_all, dtype=np.float32)

        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)


        X_train_C2 = torch.tensor(X_train_C2, dtype=torch.float32)
        X_val_C2 = torch.tensor(X_val_C2, dtype=torch.float32)
        X_test_C2 = torch.tensor(X_test_C2, dtype=torch.float32)

        X_train_all = torch.tensor(X_train_all, dtype=torch.float32)
        X_val_all = torch.tensor(X_val_all, dtype=torch.float32)
        X_test_all = torch.tensor(X_test_all, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_C2, X_train_all, y_train)
        val_dataset = TensorDataset(X_val_C2, X_val_all, y_val)
        test_dataset = TensorDataset(X_test_C2, X_test_all, y_test)

        # Create DataLoaders
        batch_size = 256
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
                    expert_outputs.append(self.experts_output[i](h))
                # expert_outputs is a list of tensors of shape (batch_size, 1)
                # Concatenate expert outputs along dimension 1
                expert_outputs = torch.cat(expert_outputs, dim=1)  # Shape: (batch_size, num_U)

                # Multiply gating_probs and expert_outputs element-wise
                weighted_expert_outputs = gating_probs * expert_outputs  # Shape: (batch_size, num_U)

                # Sum over experts (dimension 1)
                final_output = torch.sum(weighted_expert_outputs, dim=1, keepdim=True)  # Shape: (batch_size, 1)

                return final_output

        # Instantiate the model
        self.model = MixtureOfExperts(input_size_all=input_size_all, input_size_C2=input_size_C2, num_U=num_U)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 20
        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
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

            # Validation
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_C2, batch_all, batch_y in self.val_loader:
                    outputs = self.model(batch_C2, batch_all)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item() * batch_C2.size(0)
            epoch_val_loss /= len(self.val_loader.dataset)
            self.val_losses.append(epoch_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

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
        # Plot training and validation loss
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(f'./marginalizing/loss_during_training_{self.data_len}.png')

        # Plot actual vs predicted
        plt.figure(figsize=(6,6))
        plt.scatter(self.y_test_numpy, self.y_pred_test, alpha=0.5)
        plt.xlabel('Actual R')
        plt.ylabel('Predicted R')
        plt.title('Actual vs Predicted R on Test Data')
        plt.plot([0,1],[0,1], 'r--') 
        plt.savefig(f'./marginalizing/actual_vs_predicted_{self.data_len}.png')

        # Residuals
        residuals = self.y_test_numpy - self.y_pred_test
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution on Test Set')
        plt.savefig(f'./marginalizing/residuals_distribution_{self.data_len}.png')

    def plotting_data(self):
        return self.y_test_numpy, self.y_pred_test
    
    def plot_losses(self):
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(f'./marginalizing/loss_during_training_{self.data_len}.png')