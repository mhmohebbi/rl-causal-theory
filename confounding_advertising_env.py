import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import csv
import os

class AdvertisingCTR(gym.Env):
    def __init__(self):
        super(AdvertisingCTR, self).__init__()

        # Set a fixed seed for reproducibility
        self.seed()

        # Define action space: bid amounts from $0.05 to $1.00 in increments of $0.05
        self.action_space = spaces.Discrete(20)
        self.bid_values = np.linspace(0.05, 1.00, 20)

        # Define observation space: one-hot encoded user profile and contextual information
        self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.int32)

        # Advertising campaign for a women's fashion accessory targeted primarily at females aged 25-34
        # Weight parameters for click probability function
        age_weights = [0.0, 0.5, 0.0, -0.1, -0.2]  # 18-24, 25-34, 35-44, 45-54, 55+
        gender_weights = [-0.5, 0.5]               # Male, Female
        location_weights = [0.0, -0.1, 0.2, 0.2]   # North, South, East, West
        purchase_history_weights = [0.0, 0.1, 0.2, 0.3]  # No purchases, 1-5, 6-10, >10 purchases
        device_type_weights = [0.1, 0.0, 0.1]      # Mobile, Desktop, Tablet

        time_of_day_weights = [0.0, 0.0, 0.1, 0.1] # Morning, Afternoon, Evening, Night
        day_of_week_weights = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2]  # Mon-Sun
        seasonality_weights = [0.0, 0.0, 0.1, 0.1] # Spring, Summer, Fall, Winter

        self.w_s = np.concatenate([
            age_weights,
            gender_weights,
            location_weights,
            purchase_history_weights,
            device_type_weights,
            time_of_day_weights,
            day_of_week_weights,
            seasonality_weights
        ])
        self.w_a = 0.5

        # Interaction term weights
        age_interaction_weights = [0.0, 0.1, 0.0, 0.0, 0.0]
        gender_interaction_weights = [0.0, 0.1]
        location_interaction_weights = [0.0, 0.0, 0.0, 0.0]
        purchase_history_interaction_weights = [0.0, 0.0, 0.0, 0.0]
        device_type_interaction_weights = [0.0, 0.0, 0.0]
        time_of_day_interaction_weights = [0.0, 0.0, 0.0, 0.0]
        day_of_week_interaction_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        seasonality_interaction_weights = [0.0, 0.0, 0.0, 0.0]

        self.w_sa = np.concatenate([
            age_interaction_weights,
            gender_interaction_weights,
            location_interaction_weights,
            purchase_history_interaction_weights,
            device_type_interaction_weights,
            time_of_day_interaction_weights,
            day_of_week_interaction_weights,
            seasonality_interaction_weights
        ])
        self.state = None
        self.full_state = None

    def seed(self, seed=10):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32))
        self.seed(seed)

        # Sample state from state space by creating one-hot encodings for each feature
        age = np.eye(5)[np.random.choice(5)]
        gender = np.eye(2)[np.random.choice(2)]
        location = np.eye(4)[np.random.choice(4)]
        purchase_history = np.eye(4)[np.random.choice(4)]
        device_type = np.eye(3)[np.random.choice(3)]
        time_of_day = np.eye(4)[np.random.choice(4)]
        day_of_week = np.eye(7)[np.random.choice(7)]
        seasonality = np.eye(4)[np.random.choice(4)]

        # Concatenate all one-hot encoded features to form the state
        self.full_state = np.concatenate([age, gender, location, purchase_history, device_type, time_of_day, day_of_week, seasonality])
        self.state = np.concatenate([time_of_day, day_of_week, seasonality])
        return self.state, {}

    def step(self, action):
        # Get bid amount based on action index
        bid_amount = self.bid_values[action]

        # Calculate click probability using sigmoid function
        prob_click = self._calculate_click_probability(self.full_state, bid_amount)

        # Simulate click (reward is 1 if clicked, otherwise 0)
        reward = 1 if np.random.rand() < prob_click else 0

        # Each episode is one step, hence it's always done
        done = True
        info = {}
        return self.state, reward, done, False, info

    def _calculate_click_probability(self, state, bid_amount):
        interaction_term = state * bid_amount
        click_score = np.dot(self.w_s, state) + self.w_a * bid_amount + np.dot(self.w_sa, interaction_term)
        return 1 / (1 + np.exp(-click_score))
    
    def human_readable_state(self, state):
        age_categories = ['18-24', '25-34', '35-44', '45-54', '55+']
        gender_categories = ['Male', 'Female']
        location_categories = ['North', 'South', 'East', 'West']
        purchase_history_categories = ['No purchases', '1-5 purchases', '6-10 purchases', '>10 purchases']
        device_type_categories = ['Mobile', 'Desktop', 'Tablet']
        time_of_day_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
        day_of_week_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        seasonality_categories = ['Spring', 'Summer', 'Fall', 'Winter']

        age = age_categories[np.argmax(state[:5])]
        gender = gender_categories[np.argmax(state[5:7])]
        location = location_categories[np.argmax(state[7:11])]
        purchase_history = purchase_history_categories[np.argmax(state[11:15])]
        device_type = device_type_categories[np.argmax(state[15:18])]
        time_of_day = time_of_day_categories[np.argmax(state[18:22])]
        day_of_week = day_of_week_categories[np.argmax(state[22:29])]
        seasonality = seasonality_categories[np.argmax(state[29:])]

        return {
            'Age': age,
            'Gender': gender,
            'Location': location,
            'Purchase History': purchase_history,
            'Device Type': device_type,
            'Time of Day': time_of_day,
            'Day of Week': day_of_week,
            'Seasonality': seasonality
        }


# To use the environment
if __name__ == "__main__":
    env = AdvertisingCTR()
    state, _ = env.reset()
    done = False

    x = 1000
    while x > 0:
        action = env.action_space.sample()  # Randomly select an action
        state, reward, done, _, _ = env.step(action)

        file_exists = os.path.isfile('ad_ctr_data.csv')
        
        with open('ad_ctr_data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['C1', 'C2', 'Action', 'Reward'])
            writer.writerow([env.full_state, state, action, reward])

        if done:
            state, _ = env.reset()
        x -= 1