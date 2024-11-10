import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import *

def generate_data(N: int) -> pd.DataFrame:
    # Generate U
    U_samples = np.random.choice(user_types, size=N)

    # Generate C2
    C2_samples = {
        'Age': [],
        'Gender': [],
        'Location': [],
        'PurchaseHistory': [],
        'DeviceType': []
    }

    for u in U_samples:
        # Age
        age = np.random.choice(age_categories, p=age_probs_given_U[u])
        C2_samples['Age'].append(age)
        
        # Gender
        gender = np.random.choice(gender_categories, p=gender_probs_given_U[u])
        C2_samples['Gender'].append(gender)
        
        # Location
        location = np.random.choice(location_categories, p=location_probs)
        C2_samples['Location'].append(location)
        
        # Purchase History
        purchase_history = np.random.choice(purchase_history_categories, p=purchase_history_probs_given_U[u])
        C2_samples['PurchaseHistory'].append(purchase_history)
        
        # Device Type
        device_type = np.random.choice(device_type_categories, p=device_type_probs_given_U[u])
        C2_samples['DeviceType'].append(device_type)

    # Generate C1
    C1_samples = {
        'TimeOfDay': np.random.choice(time_of_day_categories, size=N, p=time_of_day_probs),
        'DayOfWeek': np.random.choice(day_of_week_categories, size=N, p=day_of_week_probs),
        'Seasonality': np.random.choice(seasonality_categories, size=N, p=seasonality_probs)
    }

    # Determine A
    A_samples = []

    for i in range(N):
        time_of_day = C1_samples['TimeOfDay'][i]
        day_of_week = C1_samples['DayOfWeek'][i]
        
        # Base bid amount
        A = 0.05  # Minimum bid
        
        # Increase bid during peak times
        if time_of_day == 'Evening':
            A += 0.20
        if day_of_week in ['Saturday', 'Sunday']:
            A += 0.15
        
        # Randomly choose a bid amount close to the calculated A
        possible_bids = action_space[action_space >= A]
        if len(possible_bids) == 0:
            A = action_space[-1]
        else:
            A = np.random.choice(possible_bids)
        
        A_samples.append(A)

    # Compute P_click and generate R
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    P_click_samples = []
    R_samples = []

    for i in range(N):
        A = A_samples[i]
        U = U_samples[i]
        theta_u = theta_U[U]
        
        g = w_A * A
        
        # Contextual features
        time_of_day = C1_samples['TimeOfDay'][i]
        g += w_C1['TimeOfDay'][time_of_day]
        
        day_of_week = C1_samples['DayOfWeek'][i]
        g += w_C1['DayOfWeek'][day_of_week]
        
        seasonality = C1_samples['Seasonality'][i]
        g += w_C1['Seasonality'][seasonality]
        
        # User profile features
        age = C2_samples['Age'][i]
        g += w_C2['Age'][age]
        
        gender = C2_samples['Gender'][i]
        g += w_C2['Gender'][gender]
        
        purchase_history = C2_samples['PurchaseHistory'][i]
        g += w_C2['PurchaseHistory'][purchase_history]
        
        device_type = C2_samples['DeviceType'][i]
        g += w_C2['DeviceType'][device_type]
        
        # Introduce random noise
        epsilon = np.random.normal(0, 0.1)
        
        # Total input to sigmoid
        total_input = g + theta_u + epsilon
        P_click = sigmoid(total_input)
        P_click_samples.append(P_click)
        
        # Generate reward
        R = np.random.binomial(1, P_click)
        R_samples.append(R)

    # Create DataFrame
    data = pd.DataFrame({
        'U': U_samples,
        'TimeOfDay': C1_samples['TimeOfDay'],
        'DayOfWeek': C1_samples['DayOfWeek'],
        'Seasonality': C1_samples['Seasonality'],
        'Age': C2_samples['Age'],
        'Gender': C2_samples['Gender'],
        'Location': C2_samples['Location'],
        'PurchaseHistory': C2_samples['PurchaseHistory'],
        'DeviceType': C2_samples['DeviceType'],
        'A': A_samples,
        'P_click': P_click_samples,
        'R': R_samples
    })

    data.to_csv(f'./data/data_{N}.csv', index=False)

    data["R"] = data["P_click"]
    data_observed = data.drop(columns=['U', 'P_click'])

    print(data_observed.head())

    # For analysis, check the distribution of P_click
    plt.clf()
    plt.hist(P_click_samples, bins=50)
    plt.title('Distribution of P_click')
    plt.xlabel('P_click')
    plt.ylabel('Frequency')
    plt.savefig(f'./p_clicks/P_click_{N}.png')
    plt.close()
    return data_observed
