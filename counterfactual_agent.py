import numpy as np

class CounterfactualAgent:
    def __init__(self, model):
        self.n_actions = 20
        self.a_values = np.linspace(0.05, 1.00, 20)

        self.model = model

    def select_action(self, context):
        """
        Selects an action based on the current context using the counterfactual policy.
        """
        # Generate counterfactual outcomes for each action
        counterfactual_rewards = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            pass
            # counterfactual_rewards[a] = self.model.counterfactual_outcome(context, 

        # Select the action with the highest counterfactual reward
        action = np.argmax(counterfactual_rewards)
        return action

"""
Could use CF on an offline dataset to train DQN
CF would generate the reward for each action it seelcts
Test it on online data for a warm start

Could use CF to generate a dataset and compare it to the original dataset for performance


"""