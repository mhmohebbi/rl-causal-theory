import numpy as np

# User types
user_types = [1, 2, 3, 4]
theta_U = {1: -0.2, 2: -0.1, 3: 0.2, 4: 0.4}

# Action space (bid amounts)
action_space = np.round(np.arange(0.05, 1.05, 0.05), 2)

# Weights
w_A = 0.75

w_C1 = {
    'TimeOfDay': {'Morning': -0.1, 'Afternoon': 0.0, 'Evening': 0.2, 'Night': -0.1},
    'DayOfWeek': {'Monday': 0.0, 'Tuesday': 0.0, 'Wednesday': 0.0, 'Thursday': 0.0,
                'Friday': 0.0, 'Saturday': 0.1, 'Sunday': 0.1},
    'Seasonality': {'Spring': 0.0, 'Summer': 0.1, 'Fall': -0.1, 'Winter': -0.2}
}

w_C2 = {
    'Age': {'18-24': 0.1, '25-34': 0.05, '35-44': 0.0, '45-54': -0.05, '55+': -0.2},
    'Gender': {'Male': 0.05, 'Female': -0.05},
    'PurchaseHistory': {'No purchases': -0.3, '1-5 purchases': -0.1, '6-10 purchases': 0.1, '10+ purchases': 0.3},
    'DeviceType': {'Mobile': 0.3, 'Desktop': 0.0, 'Tablet': -0.2}
}

# Probabilities for C2 given U 
age_categories = ['18-24', '25-34', '35-44', '45-54', '55+']
age_probs_given_U = {
    1: [0.4, 0.35, 0.15, 0.05, 0.05],
    2: [0.2, 0.3, 0.25, 0.15, 0.1],
    3: [0.1, 0.2, 0.25, 0.25, 0.2],
    4: [0.25, 0.25, 0.25, 0.15, 0.10]
}

gender_categories = ['Male', 'Female']
gender_probs_given_U = {
    1: [0.6, 0.4],
    2: [0.5, 0.5],
    3: [0.4, 0.6],
    4: [0.5, 0.5]
}

location_categories = ['North', 'South', 'East', 'West']
location_probs = [0.25, 0.25, 0.25, 0.25]

purchase_history_categories = ['No purchases', '1-5 purchases', '6-10 purchases', '10+ purchases']
purchase_history_probs_given_U = {
    1: [0.1, 0.3, 0.3, 0.3],
    2: [0.4, 0.4, 0.15, 0.05],
    3: [0.05, 0.15, 0.4, 0.4],
    4: [0.9, 0.1, 0.0, 0.0]
}

device_type_categories = ['Mobile', 'Desktop', 'Tablet']
device_type_probs_given_U = {
    1: [0.7, 0.2, 0.1],
    2: [0.4, 0.4, 0.2],
    3: [0.3, 0.5, 0.2],
    4: [0.5, 0.3, 0.2]
}

# C1
time_of_day_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
time_of_day_probs = [0.25, 0.25, 0.25, 0.25]

day_of_week_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_probs = [1/7]*7

seasonality_categories = ['Spring', 'Summer', 'Fall', 'Winter']
seasonality_probs = [0.25, 0.25, 0.25, 0.25]

list_of_features = ['TimeOfDay', 'DayOfWeek', 'Seasonality', 'Age', 'Gender', 'Location', 'PurchaseHistory', 'DeviceType']
feature_categories = {
    'TimeOfDay': time_of_day_categories,
    'DayOfWeek': day_of_week_categories,
    'Seasonality': seasonality_categories,
    'Age': age_categories,
    'Gender': gender_categories,
    'Location': location_categories,
    'PurchaseHistory': purchase_history_categories,
    'DeviceType': device_type_categories
}

feature_category_to_index = {
    'Morning': 0,
    'Afternoon': 1,
    'Evening': 2,
    'Night': 3,
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6,
    'Spring': 0,
    'Summer': 1,
    'Fall': 2,
    'Winter': 3,
    '18-24': 0,
    '25-34': 1,
    '35-44': 2,
    '45-54': 3,
    '55+': 4,
    'Male': 0,
    'Female': 1,
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'No purchases': 0,
    '1-5 purchases': 1,
    '6-10 purchases': 2,
    '10+ purchases': 3,
    'Mobile': 0,
    'Desktop': 1,
    'Tablet': 2
}