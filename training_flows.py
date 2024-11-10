import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import OneHotCategorical, Normal
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from constants import feature_category_to_index

class NormalizingFlowsTrainer:
    def __init__(self, data):
        self.data = data
        self.data_len = len(data)
        self.generative_net = None
        self.inference_net = None
        self.optimizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_dim = None
        self.num_U = 4  # Number of discrete latent classes
        self.prior_U = OneHotCategorical(probs=torch.ones(self.num_U) / self.num_U)
        self.r_recon = None
        self.training_losses = []

    def preprocess_data(self):
        categorical_columns = ['TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender', 'Location', 'PurchaseHistory', 'DeviceType']

        for col in categorical_columns:
            self.data[col] = self.data[col].apply(lambda x: feature_category_to_index[x])

        X = torch.tensor(self.data.drop(columns=['R']).values, dtype=torch.float32)
        R = torch.tensor(self.data['R'].values, dtype=torch.float32).unsqueeze(1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, R, test_size=0.2, random_state=42)
        self.X_dim = X.shape[1]

        return X, R

    def _get_realnvp_model(self):

        # Define the subnet for the U -> R flow (Coupling Layers)
        def subnet_fc(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, 256),
                nn.ReLU(),
                nn.Linear(256, d_out)
            )

        # Total input dimension: X + U
        input_dim = self.X_dim + self.num_U

        nodes = [Ff.InputNode(input_dim, name='input')]
        for k in range(3):  # Number of coupling layers
            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                                name=f'coupling_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed': k},
                                name=f'permute_{k}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        return Ff.ReversibleGraphNet(nodes, verbose=False)

    def train_model(self):
        # Variational Inference Components
        class InferenceNetwork(nn.Module):
            def __init__(self, input_dim, num_U):
                super(InferenceNetwork, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim + 1, 128),  # Input: X and R
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_U)
                )
            
            def forward(self, x, r):
                input = torch.cat([x, r], dim=1)
                logits = self.network(input)
                return logits  # Will be used with Gumbel-Softmax

        self.inference_net = InferenceNetwork(self.X_dim, self.num_U)
        self.generative_net = self._get_realnvp_model()
        self.inference_net.train()
        self.generative_net.train()

        optimizer = optim.Adam(list(self.generative_net.parameters()) + list(self.inference_net.parameters()), lr=1e-3)

        # Loss function: Negative ELBO
        def loss_function(r_recon, r, kl_divergence):
            reconstruction_loss = nn.MSELoss()(r_recon, r)
            return reconstruction_loss + kl_divergence

        num_epochs = 20
        batch_size = 64
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            total_loss = 0
            for x_batch, r_batch in data_loader:
                optimizer.zero_grad()
                
                # Inference Network: Compute logits for q(U|X,R)
                q_logits = self.inference_net(x_batch, r_batch)
                
                # Compute q(U|X,R) probabilities
                q_probs = nn.functional.softmax(q_logits, dim=1)
                # Compute log probabilities
                q_log_probs = nn.functional.log_softmax(q_logits, dim=1)
                # Prior log probabilities
                prior_log_probs = torch.log(self.prior_U.probs)

                # Compute KL divergence
                kl_divergence = torch.sum(q_probs * (q_log_probs - prior_log_probs), dim=1).mean()

                # Concatenate X and U
                x_u = torch.cat([x_batch, q_probs], dim=1)
                
                # Forward pass through RealNVP
                z, log_jac_det = self.generative_net(x_u)
                
                # Reconstruction of R (Assuming R = sigmoid(z))
                r_recon = torch.sigmoid(z[:, 0].unsqueeze(1))
                
                # Compute loss
                loss = loss_function(r_recon, r_batch, kl_divergence)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            self.training_losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    def _infer_U(self, x, r):
        # Use the trained inference network to get q(U|X,R)
        q_logits = self.inference_net(x, r)
        q_probs = nn.functional.softmax(q_logits, dim=1)
        # Choose the most probable U
        u_inferred = torch.argmax(q_probs, dim=1)
        return u_inferred, q_probs

    def evaluate_model(self):
        self.inference_net.eval()
        self.generative_net.eval()
        with torch.no_grad():
            u_inferred, u_probs = self._infer_U(self.X_test, self.y_test)
            x_u = torch.cat([self.X_test, u_probs], dim=1)
            z, _ = self.generative_net(x_u)
            r_recon = torch.sigmoid(z[:, 0].unsqueeze(1))
            self.r_recon = r_recon
            mse = mean_squared_error(self.y_test, r_recon)
            r2 = r2_score(self.y_test, r_recon)
            print(f'MSE: {mse:.4f}, R^2: {r2:.4f}')

            return mse, r2
    
    def plot_results(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.r_recon, alpha=0.5)
        plt.xlabel('Actual R')
        plt.ylabel('Reconstructed R')
        plt.title('Actual vs. Reconstructed R on Test Set')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.savefig(f'./flows/actual_vs_reconstructed_{self.data_len}.png')

        plt.figure(figsize=(8, 6))
        plt.plot(self.training_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'./flows/training_loss_{self.data_len}.png')

    def counterfactual_outcome(self, x, r, a_prime):
        u_inferred, u_probs = self._infer_U(x, r)
        x[0][-1] = a_prime # intervention
        x_u = torch.cat([x, u_probs], dim=1)
        z, _ = self.generative_net(x_u)
        r_prime = torch.sigmoid(z[:, 0].unsqueeze(1))
        return r_prime
