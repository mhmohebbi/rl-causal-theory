from data import AbstractDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from ucimlrepo import fetch_ucirepo
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import torch
from pmlb import dataset_names, fetch_data

p1 = 0.05
p2 = 0.1

class WineQualityDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=186)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="WineQuality", X=X, y=y)

    def preprocess(self):
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed
    
    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        # print("Dataset Size Pre Intervention: ", X_train.shape)
        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)

        X_intervention = torch.empty(0, X_train.shape[1])

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()
            # original_X = self.inverse_transform_x(X_intervention_copy.reshape(1, -1)).flatten()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
                # X_intervention_copy = torch.tensor(self.transform_x(original_X.reshape(1, -1)), dtype=torch.float64)
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
                # X_intervention_copy = torch.tensor(self.transform_x(original_X.reshape(1, -1)), dtype=torch.float64)
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
                # X_intervention_copy = torch.tensor(self.transform_x(original_X.reshape(1, -1)), dtype=torch.float64)
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
                # X_intervention_copy = torch.tensor(self.transform_x(original_X.reshape(1, -1)), dtype=torch.float64)
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        # print("Dataset Size Post Intervention: ", augmented_X.shape)
        return X_intervention

class AbaloneDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=1)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="Abalone", X=X, y=y)
    
    def preprocess(self):
        categorical_features = ["Sex"]
        numerical_features = ["Length", "Diameter", "Height", "Whole_weight",
                            "Shucked_weight", "Viscera_weight", "Shell_weight"]
        
        # Scale numerical features
        scaler = StandardScaler()
        X_num = scaler.fit_transform(self.X[numerical_features])
        
        # One-hot encode the categorical feature
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(self.X[categorical_features])
        
        # Combine numerical and categorical features
        X_processed = np.hstack((X_num, X_cat))

        self.X_preprocessed = X_processed
        self.y_preprocessed = self.y.values
        
        return  self.X_preprocessed, self.y_preprocessed
    
    def intervention(self, X_train):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_preprocessed = torch.tensor(X_train, dtype=torch.float64)

        mapping = list(range(X_preprocessed.shape[0]))
        X_intervention = X_preprocessed.clone()
        print("Dataset Size Pre Intervention: ", X_intervention.shape)
        for i in range(X_intervention.shape[0]):
            if X_intervention[i, -3] == 1 and X_intervention[i, -2] == 0 and X_intervention[i, -1] == 0:
                X_intervention_copy = X_intervention[i].clone()
                X_intervention_copy[-3] = 0
                X_intervention_copy[-2] = 0
                X_intervention_copy[-1] = 1
                X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
                mapping.append(i)
                X_intervention_copy[-3] = 0
                X_intervention_copy[-2] = 1
                X_intervention_copy[-1] = 0
                X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
                mapping.append(i)  # Counterfactual from sample i

            elif X_intervention[i, -3] == 0 and X_intervention[i, -2] == 1 and X_intervention[i, -1] == 0:
                X_intervention_copy = X_intervention[i].clone()
                X_intervention_copy[-3] = 1
                X_intervention_copy[-2] = 0
                X_intervention_copy[-1] = 0
                X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
                mapping.append(i)

                X_intervention_copy[-3] = 0
                X_intervention_copy[-2] = 0
                X_intervention_copy[-1] = 1
                X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
                mapping.append(i)

            elif X_intervention[i, -3] == 0 and X_intervention[i, -2] == 0 and X_intervention[i, -1] == 1:
                X_intervention_copy = X_intervention[i].clone()
                X_intervention_copy[-3] = 1
                X_intervention_copy[-2] = 0
                X_intervention_copy[-1] = 0
                X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
                mapping.append(i)

                X_intervention_copy[-3] = 0
                X_intervention_copy[-2] = 1
                X_intervention_copy[-1] = 0
                X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
                mapping.append(i)

        print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention, mapping

class ParkinsonsTelemonitoringDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=189)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="ParkinsonsTelemonitoring", X=X, y=y)

    def preprocess(self):
        self.y = self.y.drop(columns=['motor_UPDRS'])

        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed
    
    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)
        
        original_training_size = X_train.shape[0]

        # print("Dataset Size Pre Intervention: ", X_intervention.shape)

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)


        # print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention
        
class AuctionVerificationDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=713)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="AuctionVerification", X=X, y=y)
    
    def preprocess(self):
        self.y = self.y.drop(columns=['verification.result'])

        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed
    
    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])
        # print("Dataset Size Pre Intervention: ", X_intervention.shape)

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)


        # print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention
    
class RealEstateDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=477)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="RealEstate", X=X, y=y)
    
    def preprocess(self):        
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed
    
    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])
        # print("Dataset Size Pre Intervention: ", X_intervention.shape)

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        # print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention

class AirfoilDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=291)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="Airfoil Self-Noise", X=X, y=y)
    
    def preprocess(self):        
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed

    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        # print("Dataset Size Pre Intervention: ", X_intervention.shape)

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])
        
        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

            # X_intervention_copy[0] += 0.01
            # X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            # # randomly select j in range of 0 to original_training_size
            # j = np.random.randint(0, original_training_size)
            # mapping.append(i)

            # X_intervention_copy[0] -= 0.02
            # X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            # j = np.random.randint(0, original_training_size)
            # mapping.append(i)

        # print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention
    
class ConcreteCompressiveStrengthDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=165)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="ConcreteCompressiveStrength", X=X, y=y)
    
    def preprocess(self):        
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed
    
    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])
        # print("Dataset Size Pre Intervention: ", X_intervention.shape)

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        # print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention
    
class SuperconductivtyDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=464)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="Superconductivty", X=X, y=y)

    def preprocess(self):        
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed
    
    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        return X_intervention
    
    
class DailyDemandForecastingOrdersDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=409)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="DailyDemandForecastingOrders", X=X, y=y)

    def preprocess(self):
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed

    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        return X_intervention
    
class PollenDataset(AbstractDataset):
    def __init__(self):
        df = fetch_data('529_pollen')
        # how can I make y not a series but a dataframe?
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        super().__init__(name="529_pollen", X=X, y=y)

    def preprocess(self):
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed

    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        return X_intervention
    

class WindDataset(AbstractDataset):
    def __init__(self):
        df = fetch_data('503_wind')
        # how can I make y not a series but a dataframe?
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        super().__init__(name="503_wind", X=X, y=y)

    def preprocess(self):
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X.values)
        
        # Normalize the features
        self.normalizer_X = MinMaxScaler()
        self.X_preprocessed = self.normalizer_X.fit_transform(self.X_preprocessed)
        
        # Standardize the target
        self.scaler_y = StandardScaler()
        self.y_preprocessed = self.scaler_y.fit_transform(self.y.values)
        
        # Normalize the target
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y_preprocessed)

        return self.X_preprocessed, self.y_preprocessed

    def intervention(self, X_train, feature_a):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."

        X_train = torch.tensor(X_train, dtype=torch.float64)

        original_training_size = X_train.shape[0]

        half = int(original_training_size / 2)
        quarter = int(original_training_size / 4)
        three_quarters = int(3 * original_training_size / 4)
        X_intervention = torch.empty(0, X_train.shape[1])

        for i in range(original_training_size):
            X_intervention_copy = X_train[i].clone()

            if i < quarter:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] += increment
            elif i < half:
                increment = X_intervention_copy[feature_a] * p1
                X_intervention_copy[feature_a] -= increment
            elif i < three_quarters:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] += increment
            else:
                increment = X_intervention_copy[feature_a] * p2
                X_intervention_copy[feature_a] -= increment
            
            if X_intervention_copy[feature_a] < 0:
                X_intervention_copy[feature_a] = 0
            elif X_intervention_copy[feature_a] > 1:
                X_intervention_copy[feature_a] = 1

            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)

        return X_intervention
# df = kagglehub.load_dataset(
#     KaggleDatasetAdapter.PANDAS,
#     "quantbruce/real-estate-price-prediction",
#     "Real estate.csv",
#     # Provide any additional arguments like 
#     # sql_query or pandas_kwargs. See the 
#     # documenation for more information:
#     # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
#     )

# # y is the last column and everything else is X
# # extract X and y
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# # print(X.head())
# # print(y.head())
# print(df.head())
# print(df.shape)