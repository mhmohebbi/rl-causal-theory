import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Define the ODE function that is conditioned on X.
# Here T (or z) is 1D and cond is the conditioning vector (features X).
class ConditionalODEFunc(nn.Module):
    def __init__(self, cond_dim, hidden_dim=32):
        super().__init__()
        # The network takes as input [z, cond] and outputs dz/dt (a scalar per sample).
        self.net = nn.Sequential(
            nn.Linear(1 + cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, t, z, cond):
        # t is a scalar (could be used if you want time-dependence, here we ignore it),
        # z has shape (batch, 1) and cond has shape (batch, cond_dim).
        # Concatenate z and cond.
        z_cond = torch.cat([z, cond], dim=1)
        return self.net(z_cond)

# Define the conditional CNF model.
# It transforms an input T into a latent z (and computes the change in log-density).
class ConditionalCNF(nn.Module):
    def __init__(self, cond_dim, hidden_dim=32):
        super().__init__()
        self.ode_func = ConditionalODEFunc(cond_dim, hidden_dim)
    
    def forward(self, T, cond, t0=0.0, t1=1.0):
        # T is the target (shape (batch, 1))
        # cond is the conditioning vector (shape (batch, cond_dim))
        aug0 = torch.cat([T, torch.zeros_like(T)], dim=1)  # [T, logp] with logp init at 0.
        t = torch.tensor([t0, t1]).float().to(T.device)
        aug_T = odeint(lambda t, aug: self.augmented_dynamics(t, aug, cond), aug0, t)
        z_T = aug_T[-1][:, 0:1]
        delta_logp = aug_T[-1][:, 1:2]
        return z_T, delta_logp

    def augmented_dynamics(self, t, aug_state, cond):
        """
        aug_state: tensor of shape (batch, 2) where the first column is z and the second is logp.
        We compute the dynamics for both.
        """
        z = aug_state[:, 0:1]  # (batch, 1)
        logp = aug_state[:, 1:2]
        # Enable gradients for z to compute the divergence.
        if not z.requires_grad:
            z = z.requires_grad_()
        dzdt = self.ode_func(t, z, cond)  # (batch, 1)
        # Compute the divergence (trace of the Jacobian) d(dz/dt)/dz.
        divergence = torch.autograd.grad(
            dzdt, z,
            grad_outputs=torch.ones_like(dzdt),
            retain_graph=True,
            create_graph=True
        )[0]
        # For a scalar z, divergence is just the derivative.
        return torch.cat([dzdt, divergence], dim=1)

    def inverse(self, z_T, cond, t0=0.0, t1=1.0):
        # Invert the flow: given latent z, recover T.
        # We'll solve the ODE backward in time.
        aug_T = torch.cat([z_T, torch.zeros_like(z_T)], dim=1)
        t = torch.tensor([t1, t0]).float().to(z_T.device)
        aug_0 = odeint(lambda t, aug: self.augmented_dynamics(t, aug, cond), aug_T, t)
        T_recon = aug_0[-1][:, 0:1]
        return T_recon
    
    def sample(self, num_samples, cond, t0=0.0, t1=1.0):
        # For each conditioning sample, draw latent variables from base distribution and invert.
        # Here, we assume cond is of shape (batch, cond_dim). We sample num_samples for each instance.
        cond_expanded = cond.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, cond.shape[-1])
        z_samples = torch.randn(cond_expanded.shape[0], 1).to(cond.device)
        T_samples = self.inverse(z_samples, cond_expanded, t0, t1)
        return T_samples, z_samples
    
def compute_loss(model, T, cond):
    """
    T: target values (batch, 1)
    cond: conditioning variables/features (batch, cond_dim)
    The base distribution is assumed to be standard normal.
    """
    # Get the transformed latent variable and the log-determinant.
    z_T, delta_logp = model(T, cond)
    # Compute log probability under the base distribution (standard Normal).
    logp_z = -0.5 * (z_T ** 2) - 0.5 * np.log(2 * np.pi)
    # Change-of-variable formula:
    logp = logp_z - delta_logp
    loss = -logp.mean()  # maximize log-likelihood → minimize negative log-likelihood
    return loss

# Example: assume you have a DataLoader that yields batches of (T, cond)
# where T is the target (shape (batch, 1)) and cond are the features X (without Z).
def train_conditional_cnf(model, optimizer, dataloader, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for T, cond in dataloader:
            T = T.float()
            cond = cond.float()
            optimizer.zero_grad()
            loss = compute_loss(model, T, cond)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

def evaluate_reconstruction(model, T, cond):
    """
    Given test targets T and conditioning features cond, compute:
      1. The latent representation z via forward pass.
      2. Reconstruct T by applying the inverse function.
    Then, compute and return the mean squared error (MSE) of reconstruction.
    """
    model.eval()
    # with torch.no_grad():
    z, _ = model(T, cond)
    T_recon = model.inverse(z, cond)
    mse = ((T - T_recon) ** 2).mean().item()
    return mse, T_recon, z

def plot_results(T_original, T_recon, z, z_reference=None):
    """
    Plot histograms and scatter plots:
      - A scatter plot of original T vs reconstructed T.
      - Histograms of the learned latent z.
      - Optionally, compare z with a reference residual noise z_reference.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.scatter(T_original.cpu().detach().numpy(), T_recon.cpu().detach().numpy(), alpha=0.5)
    plt.xlabel("Original T")
    plt.ylabel("Reconstructed T")
    plt.title("T -> z -> T Reconstruction")
    
    plt.subplot(1,2,2)
    plt.hist(z.cpu().detach().numpy(), bins=30, alpha=0.7, label="Learned z")
    if z_reference is not None:
        plt.hist(z_reference.cpu().detach().numpy(), bins=30, alpha=0.7, label="Reference Z")
    plt.legend()
    plt.title("Latent Space Distributions")
    # save the plot
    if z_reference is not None:
        plt.savefig('./CAUSAL/conditional_cnf_results_with_reference.png')
    else:
        plt.savefig('./CAUSAL/conditional_cnf_results.png')















# --- Example usage ---
# if __name__ == "__main__":
    # Suppose we have preprocessed our dataframe and have:
    #   - T: target values (as a torch tensor of shape (N, 1))
    #   - cond: conditioning features from X (as a torch tensor of shape (N, cond_dim))
    # (We have dropped Z here; it is kept for reference only.)
    
    # For illustration, let's create some dummy data:
    # N = 1000
    # cond_dim = 10  # for example, 10 features
    # T = torch.randn(N, 1)  # dummy target values
    # cond = torch.randn(N, cond_dim)  # dummy conditioning features
    
    # # Create a simple DataLoader
    # dataset = torch.utils.data.TensorDataset(T, cond)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # # Instantiate the CNF model and optimizer
    # model = ConditionalCNF(cond_dim, hidden_dim=64)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # # Train the conditional CNF
    # train_conditional_cnf(model, optimizer, dataloader, epochs=50)
    
    # Later, you could compare the learned latent z with your previously computed residual noise Z
    # (which was not used during training) to see if they align.
