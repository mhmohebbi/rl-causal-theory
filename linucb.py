import numpy as np

class LinUCBAgent:
    def __init__(self, n_actions, context_dim, alpha=1.0):
        """
        Initializes the LinUCB agent.

        Parameters:
        - n_actions (int): Number of possible actions.
        - context_dim (int): Dimension of the context vectors.
        - alpha (float): Exploration parameter.
        """
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha

        # Initialize A (d x d identity matrices) and b (zero vectors) for each action
        self.A = [np.identity(self.context_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_actions)]

    def select_action(self, context):
        """
        Selects an action based on the current context using the LinUCB algorithm.

        Parameters:
        - context (np.array): The context vector.

        Returns:
        - action (int): The index of the selected action.
        """
        p = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            # Calculate the upper confidence bound for each action
            p[a] = context @ theta_a + self.alpha * np.sqrt(context @ A_inv @ context)

        # Select the action with the highest upper confidence bound
        action = np.argmax(p)
        return action

    def update(self, action, context, reward):
        """
        Updates the model parameters after receiving a reward.

        Parameters:
        - action (int): The action that was taken.
        - context (np.array): The context vector when the action was taken.
        - reward (float): The reward received after taking the action.
        """
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

# Example usage with your Gym environment
# Make sure to replace 'YourEnv' with the actual name of your environment

# import gym
# env = gym.make('YourEnv')

# n_actions = env.action_space.n
# context_dim = env.observation_space.shape[0]
# agent = LinUCBAgent(n_actions=n_actions, context_dim=context_dim, alpha=1.0)

# for episode in range(num_episodes):
#     context = env.reset()
#     done = False
#     while not done:
#         action = agent.select_action(context)
#         next_context, reward, done, _ = env.step(action)
#         agent.update(action, context, reward)
#         context = next_context
