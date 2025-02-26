from data import AbstractDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
import torch

class WineQualityDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=186)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="WineQuality", X=X, y=y)

    def plot_size(self):
        return 10
    
    def preprocess(self):
        self.X_preprocessed = self.X.values
        self.y_preprocessed = self.y.values
        
        return  self.X_preprocessed, self.y_preprocessed
    
    def intervention(self):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_preprocessed = torch.tensor(self.X_preprocessed, dtype=torch.float32)
        
        mapping = list(range(X_preprocessed.shape[0]))
        X_intervention = X_preprocessed.clone()
        print("Dataset Size Pre Intervention: ", X_intervention.shape)
        for i in range(X_intervention.shape[0]):
            X_intervention_copy = X_intervention[i].clone()

            X_intervention_copy[1] += 0.25
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

            X_intervention_copy[1] -= 0.5
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

        print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention, mapping

class AbaloneDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=1)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="Abalone", X=X, y=y)
    
    def plot_size(self):
        return 40
    
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
    
    def intervention(self):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_preprocessed = torch.tensor(self.X_preprocessed, dtype=torch.float32)

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
    
    def plot_size(self):
        return 60
    
    def preprocess(self):
        self.y = self.y.drop(columns=['motor_UPDRS'])
        self.X_preprocessed = self.X.values
        self.y_preprocessed = self.y.values
        
        return  self.X_preprocessed, self.y_preprocessed
    
    def intervention(self):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_preprocessed = torch.tensor(self.X_preprocessed, dtype=torch.float32)
        
        mapping = list(range(X_preprocessed.shape[0]))
        X_intervention = X_preprocessed.clone()
        print("Dataset Size Pre Intervention: ", X_intervention.shape)
        for i in range(X_intervention.shape[0]):
            X_intervention_copy = X_intervention[i].clone()

            X_intervention_copy[12] += 0.01
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

            X_intervention_copy[12] -= 0.02
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

        print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention, mapping
    
class AuctionVerificationDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=713)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="AuctionVerification", X=X, y=y)
    
    def plot_size(self):
        return 50000
    
    def preprocess(self):
        self.y = self.y.drop(columns=['verification.result'])
        # scale the features
        scaler = StandardScaler()
        self.X_preprocessed = scaler.fit_transform(self.X.values)
        # scale the target column
        # self.y_preprocessed = scaler.fit_transform(self.y.values)
        self.y_preprocessed = self.y.values
        return  self.X_preprocessed, self.y_preprocessed
    
    def intervention(self):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_preprocessed = torch.tensor(self.X_preprocessed, dtype=torch.float32)

        mapping = list(range(X_preprocessed.shape[0]))
        X_intervention = X_preprocessed.clone()
        print("Dataset Size Pre Intervention: ", X_intervention.shape)
        for i in range(X_intervention.shape[0]):
            X_intervention_copy = X_intervention[i].clone()

            X_intervention_copy[0] += 0.20
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

            X_intervention_copy[0] -= 0.40
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

        print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention, mapping
    
class RealEstateDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=477)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="RealEstate", X=X, y=y)
    
    def plot_size(self):
        return 100
    
    def preprocess(self):        
        # Scale numerical features
        scaler = StandardScaler()
        self.X_preprocessed = scaler.fit_transform(self.X.values)
        self.y_preprocessed = self.y.values
        return self.X_preprocessed, self.y_preprocessed
    
    def intervention(self):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_preprocessed = torch.tensor(self.X_preprocessed, dtype=torch.float32)

        mapping = list(range(X_preprocessed.shape[0]))
        X_intervention = X_preprocessed.clone()
        print("Dataset Size Pre Intervention: ", X_intervention.shape)
        for i in range(X_intervention.shape[0]):
            X_intervention_copy = X_intervention[i].clone()

            X_intervention_copy[2] += 0.20
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

            X_intervention_copy[2] -= 0.40
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

        print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention, mapping

class AirfoilDataset(AbstractDataset):
    def __init__(self):
        dataset = fetch_ucirepo(id=291)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        super().__init__(name="Airfoil Self-Noise", X=X, y=y)
    
    def plot_size(self):
        return 5
    
    def preprocess(self):        
        scaler = StandardScaler()
        self.X_preprocessed = scaler.fit_transform(self.X.values)
        self.y_preprocessed = scaler.fit_transform(self.y.values)
        # print(self.X_preprocessed[:5])
        
        return self.X_preprocessed, self.y_preprocessed

    def intervention(self):
        assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
        assert self.X_Z is not None, "Add Z to the dataset first."
        X_preprocessed = torch.tensor(self.X_preprocessed, dtype=torch.float32)

        mapping = list(range(X_preprocessed.shape[0]))
        X_intervention = X_preprocessed.clone()
        print("Dataset Size Pre Intervention: ", X_intervention.shape)
        for i in range(X_intervention.shape[0]):
            X_intervention_copy = X_intervention[i].clone()

            X_intervention_copy[0] += 0.01
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

            X_intervention_copy[0] -= 0.02
            X_intervention = torch.cat((X_intervention, X_intervention_copy.unsqueeze(0)), dim=0)
            mapping.append(i)

        print("Dataset Size Post Intervention: ", X_intervention.shape)

        return X_intervention, mapping
    
    