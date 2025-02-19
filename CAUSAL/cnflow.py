import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

###############################################################################
# Conditional Affine Coupling Layer
###############################################################################

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, t_dim, hidden_dim, cond_dim, mask):
        """
        A single affine coupling layer.
        
        Parameters:
          - t_dim: dimensionality of target T.
          - hidden_dim: hidden size for the NN.
          - cond_dim: dimensionality of the conditioning vector F.
          - mask: a list or tensor of 0s and 1s of length t_dim.
            The layer transforms the components where mask==1.
        """
        super(ConditionalAffineCoupling, self).__init__()
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        # Register the mask as a buffer so it is moved with the model.
        self.register_buffer('mask_tensor', torch.tensor(mask, dtype=torch.float32))
        num_masked = int(self.mask_tensor.sum().item())
        num_unmasked = t_dim - num_masked

        # The NN takes the unmasked part of T plus the conditioning F.
        nn_input_dim = num_unmasked + cond_dim
        # It outputs scaling and translation parameters for the masked part.
        nn_output_dim = num_masked * 2

        self.net = nn.Sequential(
            nn.Linear(nn_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nn_output_dim)
        )

    def forward(self, T, cond):
        """
        Forward transformation T -> y.
        Returns: transformed T (y) and log-determinant of the Jacobian.
        """
        mask = self.mask_tensor.bool()
        inv_mask = ~mask
        T_unmasked = T[:, inv_mask]
        # Concatenate the unmasked T with conditioning F.
        h = torch.cat([T_unmasked, cond], dim=1)
        params = self.net(h)
        s, t_shift = params.chunk(2, dim=1)
        # Squash s for stability.
        s = torch.tanh(s)
        T_masked = T[:, mask]
        y_masked = T_masked * torch.exp(s) + t_shift
        # Replace the masked parts.
        y = T.clone()
        y[:, mask] = y_masked
        # Compute log-determinant (only the scaling part contributes).
        log_det = torch.sum(s, dim=1)
        return y, log_det

    def inverse(self, y, cond):
        """
        Inverse transformation y -> T.
        Returns: original T and the log-determinant of the inverse Jacobian.
        """
        mask = self.mask_tensor.bool()
        inv_mask = ~mask
        y_unmasked = y[:, inv_mask]
        h = torch.cat([y_unmasked, cond], dim=1)
        params = self.net(h)
        s, t_shift = params.chunk(2, dim=1)
        s = torch.tanh(s)
        y_masked = y[:, mask]
        T_masked = (y_masked - t_shift) * torch.exp(-s)
        T = y.clone()
        T[:, mask] = T_masked
        log_det = -torch.sum(s, dim=1)
        return T, log_det

###############################################################################
# Conditional Normalizing Flow
###############################################################################

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, t_dim, cond_dim, hidden_dim=128, num_coupling_layers=6):
        """
        A normalizing flow that maps T to a latent z, conditioned on F.
        
        Parameters:
          - t_dim: Dimensionality of the target T.
          - cond_dim: Dimensionality of the conditioning features F.
          - hidden_dim: Hidden layer size for the coupling layers.
          - num_coupling_layers: Number of affine coupling layers to use.
        """
        super(ConditionalNormalizingFlow, self).__init__()
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        self.num_coupling_layers = num_coupling_layers

        # Create a series of coupling layers.
        self.layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            # Alternate masks across layers.
            if i % 2 == 0:
                mask = [1 if j < t_dim // 2 else 0 for j in range(t_dim)]
            else:
                mask = [0 if j < t_dim // 2 else 1 for j in range(t_dim)]
            self.layers.append(ConditionalAffineCoupling(t_dim, hidden_dim, cond_dim, mask))

        # Base (latent) distribution (standard normal).
        self.base_dist = D.MultivariateNormal(torch.zeros(t_dim), torch.eye(t_dim))

    def forward(self, T, cond):
        """
        Maps T -> z through the flow and accumulates the log–determinant.
        """
        log_det_total = 0
        z = T
        for layer in self.layers:
            z, log_det = layer(z, cond)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z, cond):
        """
        Inverts the mapping: z -> T.
        """
        log_det_total = 0
        T = z
        for layer in reversed(self.layers):
            T, log_det = layer.inverse(T, cond)
            log_det_total = log_det_total + log_det
        return T, log_det_total

    def log_prob(self, T, cond):
        """
        Computes the log-probability log p(T | F).
        """
        z, log_det = self.forward(T, cond)
        log_prob_z = self.base_dist.log_prob(z)
        return log_prob_z + log_det

    def fit(self, train_loader, optimizer, num_epochs=50, device='cpu'):
        """
        Trains the flow using maximum likelihood.
        """
        self.to(device)
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for T_batch, F_batch in train_loader:
                T_batch = T_batch.to(device)
                F_batch = F_batch.to(device)
                optimizer.zero_grad()
                # Negative log-likelihood loss.
                log_prob = self.log_prob(T_batch, F_batch)
                loss = -torch.mean(log_prob)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * T_batch.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    def evaluate(self, data_loader, device='cpu'):
        """
        Evaluates the flow on a validation/test set.
        """
        self.to(device)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for T_batch, F_batch in data_loader:
                T_batch = T_batch.to(device)
                F_batch = F_batch.to(device)
                log_prob = self.log_prob(T_batch, F_batch)
                loss = -torch.mean(log_prob)
                total_loss += loss.item() * T_batch.size(0)
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_recon(self, T, cond):
        """
        Given test targets T and conditioning features cond, compute:
          1. The latent representation z via forward pass.
          2. Reconstruct T by applying the inverse function.
        Then, compute and return the mean squared error (MSE) of reconstruction.
        """
        self.eval()
        with torch.no_grad():
            z, _ = self.forward(T, cond)
            T_recon, _ = self.inverse(z, cond)
            mse = ((T - T_recon) ** 2).mean().item()
        return mse, T_recon, z

        # model.eval()
        # with torch.no_grad():
        #     z, _ = model(T, cond)
        #     T_recon = model.inverse(z, cond)
        #     mse = ((T - T_recon) ** 2).mean().item()
        # return mse, T_recon, z
    
    def generate(self, F, num_samples=1, device='cpu'):
        """
        Given conditioning features F, generate samples of T.
        F should be a tensor of shape (batch_size, cond_dim).
        """
        self.to(device)
        self.eval()
        with torch.no_grad():
            # Sample latent vectors from the base distribution.
            z = self.base_dist.sample((num_samples,))
            F = F.to(device)
            # For generation, F must be provided with the same batch size as z.
            T_generated, _ = self.inverse(z, F)
        return T_generated


    def plot_results(self, T_original, T_recon, z, name, z_reference=None):
        """
        Plot histograms and scatter plots:
        - A scatter plot of original T vs reconstructed T.
        - Histograms of the learned latent z.
        - Optionally, compare z with a reference residual noise z_reference.
        """
        plt.figure(figsize=(12, 5))
        # Scatter plot: Original vs Reconstructed T
        plt.subplot(1, 2, 1)
        plt.scatter(T_original.cpu().detach().numpy(), T_recon.cpu().detach().numpy(), alpha=0.5)
        plt.xlabel("Original T")
        plt.ylabel("Reconstructed T")
        plt.title("T -> z -> T Reconstruction")
        
        # Histogram: Learned latent z
        plt.subplot(1, 2, 2)
        plt.hist(z.cpu().detach().numpy(), bins=30, alpha=0.7, label="Learned z")
        if z_reference is not None:
            plt.hist(z_reference.cpu().detach().numpy(), bins=30, alpha=0.7, label="Reference Z")
        plt.legend()
        plt.title("Latent Space Distributions")
        
        # Save the plot
        if z_reference is not None:
            plt.savefig(f'./CAUSAL/results/conditional_nf_results_with_reference_{name}.png')
        else:
            plt.savefig(f'./CAUSAL/results/conditional_nf_results_{name}.png')
        plt.close()

###############################################################################
# Example Data Preprocessing and Training
###############################################################################

# def get_preprocessed_data():
#     """
#     For demonstration we create synthetic data.
    
#     Assume T is a scalar target and F has 12 features.
#     Replace or modify this with your own data-loading logic.
#     """
#     num_samples = 1000
#     f_dim = 12
#     T = torch.randn(num_samples, 1)
#     F = torch.randn(num_samples, f_dim)
#     return T, F

# def main():
#     # -----------------------
#     # Hyperparameters
#     # -----------------------
#     t_dim = 1            # Dimensionality of target T
#     f_dim = 12           # Dimensionality of features F
#     cond_dim = f_dim     # Now conditioning is only on F
#     hidden_dim = 64
#     num_coupling_layers = 4
#     batch_size = 32
#     learning_rate = 1e-3
#     num_epochs = 50

#     # -----------------------
#     # Prepare Dataset
#     # -----------------------
#     T, F = get_preprocessed_data()
#     T_train, T_test, F_train, F_test = train_test_split(T, F, test_size=0.2, random_state=42)
#     train_dataset = TensorDataset(T_train, F_train)
#     test_dataset = TensorDataset(T_test, F_test)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # -----------------------
#     # Initialize Model and Optimizer
#     # -----------------------
#     flow = ConditionalNormalizingFlow(t_dim=t_dim, cond_dim=cond_dim, 
#                                         hidden_dim=hidden_dim,
#                                         num_coupling_layers=num_coupling_layers)
#     optimizer = optim.Adam(flow.parameters(), lr=learning_rate)

#     # -----------------------
#     # Train the Flow
#     # -----------------------
#     flow.fit(train_loader, optimizer, num_epochs=num_epochs, device='cpu')

#     # -----------------------
#     # Evaluate the Flow
#     # -----------------------
#     flow.evaluate(test_loader, device='cpu')

#     # -----------------------
#     # Generation Example
#     # -----------------------
#     # For generation, we supply some conditioning features F.
#     # Here we take the first 10 examples from the test set.
#     F_gen = F_test[:10]
#     T_generated = flow.generate(F_gen, num_samples=10, device='cpu')
#     print("Generated T samples:")
#     print(T_generated)

# if __name__ == '__main__':
#     main()
