# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import OneHotCategorical, Normal
# import FrEIA.framework as Ff
# import FrEIA.modules as Fm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# from constants import feature_category_to_index

# class NormalizingFlowsTrainer:
#     def __init__(self, data, num_U=4):
#         self.data = data
#         self.data_len = len(data)
#         self.generative_net = None
#         self.inference_net = None
#         self.optimizer = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
#         self.X_dim = None
#         self.num_U = num_U  # Number of discrete latent classes
#         self.prior_U = OneHotCategorical(probs=torch.ones(self.num_U) / self.num_U)
#         self.r_recon = None
#         self.training_losses = []

#     def preprocess_data(self):
#         categorical_columns = ['TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender', 'Location', 'PurchaseHistory', 'DeviceType']

#         for col in categorical_columns:
#             self.data[col] = self.data[col].apply(lambda x: feature_category_to_index[x])

#         X = torch.tensor(self.data.drop(columns=['R']).values, dtype=torch.float32)
#         R = torch.tensor(self.data['R'].values, dtype=torch.float32).unsqueeze(1)

#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, R, test_size=0.2, random_state=42)
#         self.X_dim = X.shape[1]

#         return X, R

#     def _get_realnvp_model(self):
#         def subnet_fc(c_in, c_out):
#             return nn.Sequential(
#                 nn.Linear(c_in, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, c_out)
#             )

#         # 1) We have T_in as the main input node (dim=1).
#         t_in = Ff.InputNode(1, name='T_in')
        
#         # 2) We have a ConditionNode of shape [X_dim + num_U].
#         cond_in = Ff.ConditionNode(self.X_dim + self.num_U, name='cond_in')

#         # Prepare a list of nodes for building the graph.
#         nodes = [t_in, cond_in]

#         # 3) First coupling block, referencing t_in as input, cond_in as condition
#         n1 = Ff.Node(
#             t_in, 
#             Fm.AllInOneBlock,
#             {'subnet_constructor': subnet_fc, 'affine_clamping': 2.0},
#             conditions=cond_in,
#             name='coupling_0'
#         )
#         nodes.append(n1)

#         # 4) Second coupling block, referencing n1 as input
#         n2 = Ff.Node(
#             n1,
#             Fm.AllInOneBlock,
#             {'subnet_constructor': subnet_fc, 'affine_clamping': 2.0},
#             conditions=cond_in,
#             name='coupling_1'
#         )
#         nodes.append(n2)

#         # 5) Third coupling block
#         n3 = Ff.Node(
#             n2,
#             Fm.AllInOneBlock,
#             {'subnet_constructor': subnet_fc, 'affine_clamping': 2.0},
#             conditions=cond_in,
#             name='coupling_2'
#         )
#         nodes.append(n3)

#         # 6) Finally, an OutputNode that ends the chain
#         t_out = Ff.OutputNode(n3, name='T_out')
#         nodes.append(t_out)

#         # 7) Build the ReversibleGraphNet
#         net = Ff.ReversibleGraphNet(nodes, verbose=False)
#         return net

#     def _get_realnvp_modell(self):
#         def subnet_fc(d_in, d_out):
#             return nn.Sequential(
#                 nn.Linear(d_in, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, d_out)
#             )

#         # The flow’s INPUT is now just T (dim=1), so:
#         input_dim = 1  # T only

#         # We will pass (X, U) as a “conditioning” vector of size X_dim + num_U
#         cond_dim = self.X_dim + self.num_U

#         nodes = []
#         nodes.append(Ff.InputNode(input_dim, name='T_in'))

#         for k in range(3):  # 3 coupling layers as you had
#             # The coupling block needs to know we have a condition of dimension=cond_dim
#             nodes.append(
#                 Ff.Node(
#                     nodes[-1],
#                     Fm.GLOWCouplingBlock,
#                     {
#                         'subnet_constructor': subnet_fc, 
#                         'clamp': 2.0, 
#                         'cond_dim': cond_dim  # pass the condition dimension
#                     },
#                     name=f'coupling_{k}'
#                 )
#             )
#             # Optionally permute or not
#             nodes.append(
#                 Ff.Node(
#                     nodes[-1],
#                     Fm.PermuteRandom,
#                     {'seed': k},
#                     name=f'permute_{k}'
#                 )
#             )

#         nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        
#         return Ff.ReversibleGraphNet(nodes, verbose=False)

#     def train_model(self):
#         # Variational Inference Components
#         class InferenceNetwork(nn.Module):
#             def __init__(self, input_dim, num_U):
#                 super(InferenceNetwork, self).__init__()
#                 self.network = nn.Sequential(
#                     nn.Linear(input_dim + 1, 128),  # Input: X and R
#                     nn.ReLU(),
#                     nn.Linear(128, 64),
#                     nn.ReLU(),
#                     nn.Linear(64, num_U)
#                 )
            
#             def forward(self, x, r):
#                 input = torch.cat([x, r], dim=1)
#                 logits = self.network(input)
#                 return logits  # Will be used with Gumbel-Softmax

#         self.inference_net = InferenceNetwork(self.X_dim, self.num_U)
#         self.generative_net = self._get_realnvp_model()
#         self.inference_net.train()
#         self.generative_net.train()

#         optimizer = optim.Adam(list(self.generative_net.parameters()) + list(self.inference_net.parameters()), lr=1e-3)

#         # Loss function: Negative ELBO
#         def loss_function(r_recon, r, kl_divergence):
#             reconstruction_loss = nn.MSELoss()(r_recon, r)
#             return reconstruction_loss + kl_divergence

#         num_epochs = 20
#         batch_size = 64
#         dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#         data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         for epoch in range(num_epochs):
#             total_loss = 0.0
#             for x_batch, t_batch in data_loader:
#                 optimizer.zero_grad()

#                 # -----------------------
#                 # 1) Inference for U:
#                 # -----------------------
#                 q_logits = self.inference_net(x_batch, t_batch)  # q(U|X,T)
#                 q_probs = nn.functional.softmax(q_logits, dim=1)
#                 q_log_probs = nn.functional.log_softmax(q_logits, dim=1)
#                 prior_log_probs = torch.log(self.prior_U.probs)

#                 # KL( q(U|X,T) || p(U) )
#                 kl_divergence = torch.sum(
#                     q_probs * (q_log_probs - prior_log_probs), 
#                     dim=1
#                 ).mean()

#                 # -----------------------
#                 # 2) Flow log-likelihood for T:
#                 # -----------------------
#                 #   Condition = cat(X, q_probs) of shape (batch_size, X_dim+num_U)
#                 cond = torch.cat([x_batch, q_probs], dim=1)  
                
#                 # Forward pass: T -> z
#                 # z, log_jac_det = self.generative_net(
#                 #     t_batch,  # shape (batch_size, 1)
#                 #     c=cond     # condition
#                 # )
#                 z_list, log_jac_det = self.generative_net([t_batch], c=[cond], rev=False)
#                 z = z_list[0]

#                 # z is also shape (batch_size, 1)
#                 # negative log-likelihood = 0.5*z^2 - log|det J|
#                 nll = 0.5 * (z**2) - log_jac_det  # shape (batch_size, 1)
#                 nll = nll.mean()

#                 # -----------------------
#                 # 3) Final loss = E_q[ NLL + KL ]
#                 # -----------------------
#                 loss = nll + kl_divergence
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()

#             avg_loss = total_loss / len(data_loader)
#             self.training_losses.append(avg_loss)
#             print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')


#     def _infer_U(self, x, r):
#         # Use the trained inference network to get q(U|X,R)
#         q_logits = self.inference_net(x, r)
#         q_probs = nn.functional.softmax(q_logits, dim=1)
#         # Choose the most probable U
#         u_inferred = torch.argmax(q_probs, dim=1)
#         return u_inferred, q_probs

#     def evaluate_modell(self):
#         self.inference_net.eval()
#         self.generative_net.eval()
#         with torch.no_grad():
#             u_inferred, u_probs = self._infer_U(self.X_test, self.y_test)
#             # x_u = torch.cat([self.X_test, u_probs], dim=1)
#             # z, _ = self.generative_net(x_u)
#             x_r_u = torch.cat([self.X_test, self.y_test, u_probs], dim=1)
#             z, _ = self.generative_net(x_r_u)
#             r_recon = torch.sigmoid(z[:, 0].unsqueeze(1))
#             self.r_recon = r_recon
#             mse = mean_squared_error(self.y_test, r_recon)
#             r2 = r2_score(self.y_test, r_recon)
#             print(f'MSE: {mse:.4f}, R^2: {r2:.4f}')

#             return mse, r2
        
#     def evaluate_model(self):
#         self.inference_net.eval()
#         self.generative_net.eval()
#         with torch.no_grad():
#             # 1) Infer U
#             q_logits = self.inference_net(self.X_test, self.y_test)
#             q_probs = nn.functional.softmax(q_logits, dim=1)

#             # 2) Create the condition = [X, U]
#             cond = torch.cat([self.X_test, q_probs], dim=1)

#             # 3) We want to invert from z=0 => T
#             # z_zero = torch.zeros_like(self.y_test)
#             z_zero = torch.zeros((self.X_test.size(0), 1), dtype=torch.float32)

#             # --- INSTEAD OF self.generative_net.inverse(...) DO ---
#             # pass rev=True to do the inverse pass
#             t_pred_list, _ = self.generative_net([z_zero], c=[cond], rev=True)
#             t_pred = t_pred_list[0]  # unwrap from list
#             self.r_recon = t_pred

#             # Check shape
#             print("t_pred shape:", t_pred.shape)       # want [9525, 1]
#             print("self.y_test shape:", self.y_test.shape)  # [9525, 1]

#             # If t_pred is shape [1, 9525], transpose it:
#             if t_pred.shape[0] == 1 and t_pred.shape[1] == self.y_test.shape[0]:
#                 t_pred = t_pred.transpose(0, 1)  # shape => [9525, 1]

#             # If t_pred is just [1], something else is off (but let's just be safe):
#             if t_pred.shape == torch.Size([1]):
#                 # Make it match y_test length (this fallback shouldn't really happen if training shape is correct)
#                 t_pred = t_pred.repeat(self.y_test.size(0), 1)


#             # 4) Evaluate MSE and R^2
#             mse = mean_squared_error(self.y_test, t_pred)
#             r2 = r2_score(self.y_test, t_pred)

#             print(f'MSE: {mse:.4f}, R^2: {r2:.4f}')
#             return mse, r2


#     def plot_results(self):
#         plt.figure(figsize=(8, 6))
#         plt.scatter(self.y_test, self.r_recon, alpha=0.5)
#         plt.xlabel('Actual R')
#         plt.ylabel('Reconstructed R')
#         plt.title('Actual vs. Reconstructed R on Test Set')
#         plt.plot([0, 1], [0, 1], 'r--')
#         plt.savefig(f'./flows/actual_vs_reconstructed_{self.data_len}.png')

#         plt.figure(figsize=(8, 6))
#         plt.plot(self.training_losses)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training Loss')
#         plt.savefig(f'./flows/training_loss_{self.data_len}.png')

#     def counterfactual_outcome(self, x, r, a_prime):
#         u_inferred, u_probs = self._infer_U(x, r)
#         x[0][-1] = a_prime # intervention
#         x_u = torch.cat([x, u_probs], dim=1)
#         z, _ = self.generative_net(x_u)
#         r_prime = torch.sigmoid(z[:, 0].unsqueeze(1))
#         return r_prime
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
import torch.nn.functional as F

class NormalizingFlowsTrainer:
    def __init__(self, data, num_U=4):
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
        self.num_U = num_U  # Number of discrete latent classes
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

    #
    #  REPLACED the node-based approach with a simpler cINN built using SequenceINN
    #
    def _get_realnvp_model(self):
        def subnet_fc(c_in, c_out):
            return nn.Sequential(
                nn.Linear(c_in, 256),
                nn.ReLU(),
                nn.Linear(256, c_out)
            )
        
        # We have 1D data for T (dim=1). So we create a SequenceINN with input dimension=1
        net = Ff.SequenceINN(2)
        # We'll add 3 AllInOneBlock steps, each with a condition shape of (X_dim + num_U,)
        for _ in range(7):
            net.append(
                Fm.AllInOneBlock,
                cond_shape=(self.X_dim + self.num_U,),
                subnet_constructor=subnet_fc,
                affine_clamping=2.0
            )
        return net

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
                return logits

        self.inference_net = InferenceNetwork(self.X_dim, self.num_U)
        self.generative_net = self._get_realnvp_model()
        self.inference_net.train()
        self.generative_net.train()

        optimizer = optim.Adam(
            list(self.generative_net.parameters()) + list(self.inference_net.parameters()),
            lr=1e-5
        )

        num_epochs = 20
        batch_size = 64
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_batch, t_batch in data_loader:
                optimizer.zero_grad()

                # 1) Inference for U:
                q_logits = self.inference_net(x_batch, t_batch)  # q(U|X,T)
                q_probs = nn.functional.softmax(q_logits, dim=1)
                q_log_probs = nn.functional.log_softmax(q_logits, dim=1)
                prior_log_probs = torch.log(self.prior_U.probs)

                kl_divergence = torch.sum(
                    q_probs * (q_log_probs - prior_log_probs),
                    dim=1
                ).mean()

                # 2) Flow log-likelihood for T (forward pass, rev=False)
                # Condition is cat(X, q_probs) => shape [batch_size, X_dim+num_U]
                cond = torch.cat([x_batch, q_probs], dim=1)

                #
                # IMPORTANT: now we call the flow with .forward(...) style:
                #
                t_pad = torch.cat([t_batch, torch.zeros_like(t_batch)], dim=1)

                z, log_jac_det = self.generative_net(t_pad, c=cond, rev=False) 
                # z, log_jac_det are each shape [batch_size] or [batch_size, 1]
                # Because T is 1D

                # If z is shape (batch_size,1), do z**2 => shape (batch_size,1), sum(dim=1) => shape (batch_size)
                # then subtract log_jac_det => shape (batch_size)
                # and average
                if z.dim() == 2:
                    # sum over the 2 latent dims => shape [batch_size]
                    nll = 0.5*(z**2).sum(dim=1) - log_jac_det
                else:
                    # e.g. if it came out shape [batch_size], that would be unusual with 2D
                    nll = 0.5*(z**2) - log_jac_det
                nll = nll.mean()

                T_mean_pred, _ = self.generative_net(
                    torch.zeros_like(t_pad), 
                    c=cond, 
                    rev=True
                )
                # extract the first dimension if it's shape (batch,2)
                T_mean_pred = T_mean_pred[:,0:1]
                mse_term = F.mse_loss(T_mean_pred, t_batch[:,0:1])
            
                loss = nll + kl_divergence + 0.01*mse_term
                # loss = nll + kl_divergence
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            self.training_losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    def _infer_U(self, x, r):
        q_logits = self.inference_net(x, r)
        q_probs = nn.functional.softmax(q_logits, dim=1)
        u_inferred = torch.argmax(q_probs, dim=1)
        return u_inferred, q_probs

    def evaluate_model(self):
        self.inference_net.eval()
        self.generative_net.eval()

        with torch.no_grad():
            # 1) Infer U
            q_logits = self.inference_net(self.X_test, self.y_test)
            q_probs = nn.functional.softmax(q_logits, dim=1)

            # 2) Condition
            cond = torch.cat([self.X_test, q_probs], dim=1)  # shape [9525, X_dim+num_U]

            # 3) first go forward to get the z
            t_pad = torch.cat([self.y_test, torch.zeros_like(self.y_test)], dim=1)

            z_samples, _ = self.generative_net(t_pad, c=cond, rev=False)

            #
            # Now call the inverse pass: rev=True
            #
            # t_pred, _ = self.generative_net(z_samples, c=cond, rev=True)
            t_pad_pred, _ = self.generative_net(z_samples, c=cond, rev=True)

            # t_pred should be [9525, 1] or [9525]

            # If it’s [9525], expand it to [9525,1] so it matches y_test’s shape
            # if t_pred.dim() == 1:
            #     t_pred = t_pred.unsqueeze(1)
            t_pred = t_pad_pred[:, [0]]  # keep dim

            self.r_recon = t_pred

            # 4) Evaluate MSE & R^2
            mse = mean_squared_error(
                self.y_test.cpu().numpy(),
                t_pred.cpu().numpy()
            )
            r2 = r2_score(
                self.y_test.cpu().numpy(),
                t_pred.cpu().numpy()
            )

            print("t_pred shape:", t_pred.shape)
            print("self.y_test shape:", self.y_test.shape)
            print(f'MSE: {mse:.4f}, R^2: {r2:.4f}')
            return mse, r2

    def plot_results(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.r_recon, alpha=0.5)
        plt.xlabel('Actual T')
        plt.ylabel('Reconstructed T')
        plt.title('Actual vs. Reconstructed T on Test Set')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.savefig(f'./flows/actual_vs_reconstructed_{self.data_len}.png')

        plt.figure(figsize=(8, 6))
        plt.plot(self.training_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'./flows/training_loss_{self.data_len}.png')

    def counterfactual_outcome(self, x, r, a_prime):
        x = x.unsqueeze(0)
        r = r.unsqueeze(0)
        u_inferred, u_probs = self._infer_U(x, r)
        x_u = torch.cat([x, u_probs], dim=1)

        r_pad = torch.cat([r, torch.zeros_like(r)], dim=1)
        z, _ = self.generative_net(r_pad, c=x_u, rev=False)
        print("x is", x)
        x[0][0] = a_prime # intervention
        x[0][1] = a_prime # intervention
        x[0][2] = a_prime # intervention
        x[0][4] = a_prime # intervention

        x_u = torch.cat([x, u_probs], dim=1)
        #make x_u all zeroes now
        x_u = torch.zeros_like(x_u)
        print("x is now", x_u)

        t_prime_pad, _ = self.generative_net(z, c=x_u, rev=True)
        t_prime = t_prime_pad[:, [0]]
        return t_prime
