import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

###############################################################################
# Inference Network: Predict a distribution for U given (T, F)
###############################################################################

class InferenceNetwork(nn.Module):
    def __init__(self, t_dim, f_dim, u_dim, hidden_dim=128):
        """
        Given T (target) and F (features), predict parameters (mean, log_std)
        of a Gaussian distribution over the latent variable U.
        """
        super(InferenceNetwork, self).__init__()
        input_dim = t_dim + f_dim
        output_dim = 2 * u_dim  # we output both mean and log_std for U
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.u_dim = u_dim

    def forward(self, T, F):
        # T: (batch_size, t_dim), F: (batch_size, f_dim)
        x = torch.cat([T, F], dim=1)
        out = self.net(x)  # (batch_size, 2*u_dim)
        mean, log_std = out.chunk(2, dim=1)
        return mean, log_std

def sample_gaussian(mean, log_std):
    """Reparameterization: sample U ~ N(mean, diag(exp(log_std)^2))."""
    std = torch.exp(log_std)
    eps = torch.randn_like(std)
    return mean + eps * std

###############################################################################
# Conditional RealNVP Coupling Layer
###############################################################################

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, t_dim, hidden_dim, cond_dim, mask):
        """
        A single affine coupling layer.
        
        Parameters:
          - t_dim: dimensionality of target T.
          - hidden_dim: hidden size for the NN.
          - cond_dim: dimensionality of the conditioning vector (F and U concatenated).
          - mask: a list or tensor of 0s and 1s of length t_dim. The coupling
            layer will transform the components where mask==1 while leaving the others fixed.
        """
        super(ConditionalAffineCoupling, self).__init__()
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        # Convert mask to a tensor of floats and register as buffer.
        self.register_buffer('mask_tensor', torch.tensor(mask, dtype=torch.float32))
        num_masked = int(self.mask_tensor.sum().item())
        num_unmasked = t_dim - num_masked

        # The NN takes as input the unmasked part of T plus the conditioning vector.
        nn_input_dim = num_unmasked + cond_dim
        # It outputs scaling and translation for the masked part (2 values per masked dim)
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
        Forward transformation of T.
        Returns: transformed T (y) and log-determinant of the Jacobian.
        """
        # Select unmasked and masked parts of T.
        mask = self.mask_tensor.bool()            # indices to be transformed
        inv_mask = (~self.mask_tensor.bool())       # indices kept unchanged
        T_unmasked = T[:, inv_mask]  # (batch, num_unmasked)
        # Concatenate unmasked part with conditioning vector.
        h = torch.cat([T_unmasked, cond], dim=1)
        params = self.net(h)  # (batch, 2*num_masked)
        s, t_shift = params.chunk(2, dim=1)  # each of shape (batch, num_masked)
        # For numerical stability, squashing the scaling factors.
        s = torch.tanh(s)
        # Extract the masked part of T.
        T_masked = T[:, mask]  # (batch, num_masked)
        # Apply affine transformation.
        y_masked = T_masked * torch.exp(s) + t_shift
        # Build output by leaving unmasked parts untouched.
        y = T.clone()
        y[:, mask] = y_masked
        # The log–determinant is the sum of s over the masked dimensions.
        log_det = torch.sum(s, dim=1)
        return y, log_det

    def inverse(self, y, cond):
        """
        Inverse transformation.
        Returns: original T and the log-determinant of the inverse Jacobian.
        """
        mask = self.mask_tensor.bool()
        inv_mask = (~self.mask_tensor.bool())
        y_unmasked = y[:, inv_mask]
        h = torch.cat([y_unmasked, cond], dim=1)
        params = self.net(h)
        s, t_shift = params.chunk(2, dim=1)
        s = torch.tanh(s)
        # Invert the affine transformation.
        y_masked = y[:, mask]
        T_masked = (y_masked - t_shift) * torch.exp(-s)
        T = y.clone()
        T[:, mask] = T_masked
        log_det = -torch.sum(s, dim=1)
        return T, log_det

###############################################################################
# Conditional RealNVP Flow: Compose Multiple Coupling Layers
###############################################################################

class ConditionalRealNVP(nn.Module):
    def __init__(self, t_dim, cond_dim, hidden_dim=128, num_coupling_layers=6):
        """
        A normalizing flow that maps T to latent z, conditioned on (F, U).
        """
        super(ConditionalRealNVP, self).__init__()
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        self.num_coupling_layers = num_coupling_layers

        self.layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            # Alternate masks: for example, if t_dim=4, use [1,1,0,0] then [0,0,1,1].
            if i % 2 == 0:
                # Mask first half of dimensions.
                mask = [1 if j < t_dim // 2 else 0 for j in range(t_dim)]
            else:
                # Mask second half.
                mask = [0 if j < t_dim // 2 else 1 for j in range(t_dim)]
            self.layers.append(ConditionalAffineCoupling(t_dim, hidden_dim, cond_dim, mask))

        # Define the base (latent) distribution.
        self.base_dist = D.MultivariateNormal(torch.zeros(t_dim), torch.eye(t_dim))

    def forward(self, T, cond):
        """
        Maps T -> z and accumulates the log–determinant.
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
        Computes the log-probability of T given conditioning vector cond.
        """
        z, log_det = self.forward(T, cond)
        log_prob_z = self.base_dist.log_prob(z)
        return log_prob_z + log_det

###############################################################################
# KL Divergence between q(U|T,F) and p(U) (both Gaussian, diagonal)
###############################################################################

def kl_divergence(mean, log_std):
    # For a Gaussian with diagonal covariance and a standard normal prior:
    # KL(q||p) = 0.5 * sum(exp(2*log_std) + mean^2 - 1 - 2*log_std)
    return 0.5 * torch.sum(torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std, dim=1)

###############################################################################
# Main Training Loop Example
###############################################################################


def get_preprocessed_data():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 

    X_old = adult.data.features
    y_old = adult.data.targets

    # add y_old to X_old
    X_old['income'] = y_old

    #drop "hours-per-week" from X_old and make it y_old
    y_old = X_old['hours-per-week']
    X_old = X_old.drop(columns=['hours-per-week'])

    X = X_old.dropna()
    y = y_old.loc[X.index]
    
    # Optional checks
    assert not X.isnull().values.any(), "There are still missing values in X."
    assert not y.isnull().values.any(), "There are still missing values in y."

    # 1) One-hot encode all categorical columns in X
    X_encoded = X.copy()

    # Loop through columns to encode categorical ones
    for column in X_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X_encoded[column])

    # Convert boolean columns to integers
    X_encoded = X_encoded.astype(int)

    # drop education column
    X_encoded = X_encoded.drop(columns=['education'])
    X_encoded = X_encoded.drop(columns=['fnlwgt'])
    # scaler = StandardScaler()
    # X_encoded = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

    # 2)  encode y
    # we want to normalize the target variable
    values = y.unique()
    scaled_values = (values - values.min()) / (values.max() - values.min())
    y_encoded = y.replace(dict(zip(values, scaled_values)))

    # Optionally, check that everything is numeric now
    assert all([pd.api.types.is_numeric_dtype(dt) for dt in X_encoded.dtypes]), \
        "Not all feature columns are numeric."
    if isinstance(y_encoded, pd.DataFrame):
        assert all([pd.api.types.is_numeric_dtype(dt) for dt in y_encoded.dtypes]), \
            "Not all target columns are numeric."

    print(X_encoded.head())
    print(y_encoded.head())
    # turn them into tensors
    X_encoded = torch.tensor(X_encoded.values, dtype=torch.float32)
    y_encoded = torch.tensor(y_encoded.values, dtype=torch.float32).unsqueeze(1)
    return X_encoded, y_encoded

def main():
    # -----------------------
    # Hyperparameters
    # -----------------------
    t_dim = 1   # dimensionality of target T (you can adjust)
    f_dim = 12   # dimensionality of feature F
    u_dim = 1   # dimensionality of latent noise U
    cond_dim = f_dim + u_dim  # conditioning will be the concatenation of F and U
    hidden_dim = 64
    num_coupling_layers = 2
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50

    # -----------------------
    # Instantiate Networks
    # -----------------------
    flow = ConditionalRealNVP(t_dim=t_dim, cond_dim=cond_dim, hidden_dim=hidden_dim,
                                num_coupling_layers=num_coupling_layers)
    inference_net = InferenceNetwork(t_dim=t_dim, f_dim=f_dim, u_dim=u_dim, hidden_dim=hidden_dim)

    # Combine parameters from both networks.
    optimizer = optim.Adam(list(flow.parameters()) + list(inference_net.parameters()), lr=learning_rate)

    # -----------------------
    # Create Dummy Dataset (Replace with your own data)
    # -----------------------
    X, y = get_preprocessed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_data = X_train.size(0)
    dataset = TensorDataset(y_train, X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -----------------------
    # Training Loop
    # -----------------------
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for T_batch, F_batch in dataloader:
            # -----------------------
            # 1. Inference Network: Predict distribution q(U|T,F)
            # -----------------------
            mean, log_std = inference_net(T_batch, F_batch)
            U_sample = sample_gaussian(mean, log_std)  # sample U via reparameterization

            # -----------------------
            # 2. Flow: Compute log likelihood for T given condition (F, U)
            # -----------------------
            # The condition for the flow is the concatenation of F and the sampled U.
            cond = torch.cat([F_batch, U_sample], dim=1)
            log_prob = flow.log_prob(T_batch, cond)  # log p(T|F, U)
            nll_loss = -torch.mean(log_prob)

            # -----------------------
            # 3. KL Divergence: Regularize q(U|T,F) towards the standard normal prior.
            # -----------------------
            kl_loss = torch.mean(kl_divergence(mean, log_std))

            # -----------------------
            # 4. Total Loss: Negative ELBO (we maximize ELBO so minimize its negative)
            # -----------------------
            loss = nll_loss + kl_loss

            # -----------------------
            # 5. Backpropagation and Optimization
            # -----------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * T_batch.size(0)

        epoch_loss = running_loss / num_data
        losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # -----------------------
    # Example: Inverse Pass
    # -----------------------
    # Given a latent sample z and condition, you can generate a T sample:
    z_sample = flow.base_dist.sample((10,))  # sample 10 latent vectors
    # For generation, you need to supply a conditioning vector. For example, set F to zeros and U to zeros:
    F_gen = torch.zeros(10, f_dim)
    U_gen = torch.zeros(10, u_dim)
    cond_gen = torch.cat([F_gen, U_gen], dim=1)
    T_generated, _ = flow.inverse(z_sample, cond_gen)
    print("Generated T samples:", T_generated)
    print()

    # -----------------------
    # Example: Evaluation
    # -----------------------
    inference_net.eval()
    flow.eval()
    
    # save the models
    torch.save(inference_net.state_dict(), './inference_net.pth')
    torch.save(flow.state_dict(), './flow.pth')

    reconstructions = []
    with torch.no_grad():
        mean, log_std = inference_net(y_test, X_test)
        U_sample = sample_gaussian(mean, log_std)
        cond = torch.cat([X_test, U_sample], dim=1)
        z, _ = flow(y_test, cond)

        # change the first element of cond to be --1
        # for i in range(len(cond)):
        #     cond[i][0] += -10
        
        t_reconstructed, _ = flow.inverse(z, cond)
        reconstructions.append(t_reconstructed)
    mse = mean_squared_error(y_test, torch.cat(reconstructions).numpy())
    r2 = r2_score(y_test, torch.cat(reconstructions).numpy())
    print(f"Test MSE: {mse:.4f}, R^2: {r2:.4f}")

    base_model = base_train_model(X_train, y_train)
    base_evaluate_model(base_model, X_test, y_test)

    reconstructions = []
    new_t = []
    new_x = []
    with torch.no_grad():
        mean, log_std = inference_net(y_test, X_test)
        U_sample = sample_gaussian(mean, log_std)
        cond = torch.cat([X_test, U_sample], dim=1)
        z, _ = flow(y_test, cond)

        # change the first element of cond to be --1
        for transformation in (-10, -5, 5, 10):
            # make a copy of X_test to avoid changing the original
            temp_X_test = X_test.clone()
            for i in range(len(X_test)):
                temp_X_test[i][0] += transformation

            cond = torch.cat([temp_X_test, U_sample], dim=1)
            t_prime, _ = flow.inverse(z, cond)
            new_x.append(temp_X_test)
            new_t.append(t_prime)

        # reconstructions.append(t_reconstructed)
    # merge all of the new_x into one array, so if it is like [[1,2,3], [4,5,6], [7,8,9]] it becomes [1,2,3,4,5,6,7,8,9]
    new_x = torch.cat(new_x)
    new_t = torch.cat(new_t)
    # add them to X train and y train
    X_train = torch.cat((X_train, new_x))
    y_train = torch.cat((y_train, new_t))

    base_model = base_train_model(X_train, y_train)
    base_evaluate_model(base_model, X_test, y_test, True)

    # mse = mean_squared_error(y_test, torch.cat(reconstructions).numpy())
    # r2 = r2_score(y_test, torch.cat(reconstructions).numpy())
    # print(f"Test MSE: {mse:.4f}, R^2: {r2:.4f}")
    # plot_results()

    def plot_results():
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, reconstructions, alpha=0.5)
        plt.xlabel('Actual T')
        plt.ylabel('Reconstructed T')
        plt.title('Actual vs. Reconstructed T on Test Set')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.savefig(f'./test.png')

        plt.figure(figsize=(8, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'./loss.png')
        plt.close()

def base_train_model(X_train, y_train):
    # model = MLPRegressor(
    #     hidden_layer_sizes=(32, 32),
    #     activation='relu',
    #     solver='adam',
    #     max_iter=100,
    #     batch_size=64,
    #     learning_rate='constant',
    #     learning_rate_init=0.0001,
    #     tol=0.0,
    #     n_iter_no_change=50,
    #     alpha=0.0, 
    #     verbose=True,
    #     random_state=10
    # )
    model = xgb.XGBRegressor(objective='reg:squarederror',
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,

                            seed=42)

    model.fit(X_train, y_train)
    return model

def base_evaluate_model(model, X_test, y_test, second=False):
    y_pred_test = model.predict(X_test)

    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f'Testing MSE: {mse_test:.4f}, R^2: {r2_test:.4f}')

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.xlabel('Actual T')
    plt.ylabel('Predicted T')
    plt.title('Actual vs. Predicted T on Test Set')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.savefig(f'./base{second}.png')
    plt.close()
    return mse_test, r2_test

if __name__ == '__main__':
    main()
