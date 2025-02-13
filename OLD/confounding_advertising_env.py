import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pickle
from stable_baselines3.common.buffers import ReplayBuffer
from constants import *


class AdvertisingEnv(gym.Env):
    def __init__(self, save_buffer=False, buffer_size=2000):
        super(AdvertisingEnv, self).__init__()

        # Set a fixed seed for reproducibility
        self.seed()

        # Define action space: bid amounts from $0.05 to $1.00 in increments of $0.05
        self.action_space = spaces.Discrete(20)
        self.bid_values = np.linspace(0.05, 1.00, 20)

        # Define observation space: user profile and contextual information
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        self.U = None
        self.state = None

        self.save_buffer = save_buffer
        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(self.buffer_size, observation_space=self.observation_space, action_space=self.action_space)
        self.buffer_saved = False

    def sample(self):
        u = np.random.choice(user_types)
        age = np.random.choice(age_categories, p=age_probs_given_U[u])
        gender = np.random.choice(gender_categories, p=gender_probs_given_U[u])
        location = np.random.choice(location_categories, p=location_probs)
        purchase_history = np.random.choice(purchase_history_categories, p=purchase_history_probs_given_U[u])
        device_type = np.random.choice(device_type_categories, p=device_type_probs_given_U[u])

        # Generate C1
        time_of_day_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
        time_of_day_probs = [0.25, 0.25, 0.25, 0.25]

        day_of_week_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week_probs = [1/7]*7

        seasonality_categories = ['Spring', 'Summer', 'Fall', 'Winter']
        seasonality_probs = [0.25, 0.25, 0.25, 0.25]

        time_of_day = np.random.choice(time_of_day_categories, p=time_of_day_probs)
        day_of_week = np.random.choice(day_of_week_categories, p=day_of_week_probs)
        seasonality = np.random.choice(seasonality_categories, p=seasonality_probs)

        self.U = u
        return np.array([
            feature_category_to_index[time_of_day],
            feature_category_to_index[day_of_week],
            feature_category_to_index[seasonality],
            feature_category_to_index[age],
            feature_category_to_index[gender],
            feature_category_to_index[location],
            feature_category_to_index[purchase_history],
            feature_category_to_index[device_type]
        ], dtype=np.float32)
            

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=0):
        self.seed(seed)
        self.state = self.sample()
        return self.state, {}

    def step(self, action):
        # Get bid amount based on action index
        bid_amount = self.bid_values[action]

        # Calculate click probability using sigmoid function
        reward = self._calculate_click_probability(self.state, bid_amount)

        # Each episode is one step, hence it's always done
        done = True
        info = {}

        # Collect data
        if self.save_buffer:
            self.buffer.add(self.state, self.state, action, reward, done, [{}]) # update buffer for offline RL with pkls
            if self.buffer.full and not self.buffer_saved:
                print("Buffer full, saving to .pkl file")
                with open(f"./offline_dqn/data/advertising_scm_{self.buffer_size}.pkl", 'wb') as file:
                    pickle.dump(self.buffer, file)
                self.buffer_saved = True

        return self.state, reward, done, False, info

    def _calculate_click_probability(self, context, bid_amount):
        theta_u = theta_U[self.U]
        g = w_A * bid_amount
        
        g += w_C1['TimeOfDay'][feature_categories['TimeOfDay'][int(context[0])]]
        g += w_C1['DayOfWeek'][feature_categories['DayOfWeek'][int(context[1])]]
        g += w_C1['Seasonality'][feature_categories['Seasonality'][int(context[2])]]

        g += w_C2['Age'][feature_categories['Age'][int(context[3])]]
        g += w_C2['Gender'][feature_categories['Gender'][int(context[4])]]
        g += w_C2['Location'][feature_categories['Location'][int(context[5])]]
        g += w_C2['PurchaseHistory'][feature_categories['PurchaseHistory'][int(context[6])]]
        g += w_C2['DeviceType'][feature_categories['DeviceType'][int(context[7])]]

        epsilon = np.random.normal(0, 0.1)
        click_score = g + theta_u + epsilon
        return 1 / (1 + np.exp(-click_score)) # Sigmoid function
    
# # To use the environment
if __name__ == "__main__":
    env = AdvertisingEnv(True, 3500)
    state, _ = env.reset()
    done = False

    x = 4600
    while x > 0:
        action = env.action_space.sample()  # Randomly select an action
        state, reward, done, _, _ = env.step(action)
        print(f"State: {state}, Reward: {reward}")
        if done:
            state, _ = env.reset()
        x -= 1