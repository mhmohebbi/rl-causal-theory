from data import AbstractDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from ucimlrepo import fetch_ucirepo
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import torch
from pmlb import dataset_names, fetch_data


def get_large():
    df = fetch_data('564_fried')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    return X, y

DATASETS = []


df = fetch_data('564_fried')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="564_fried", X=X, y=y))

df = fetch_data('595_fri_c0_1000_10')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="595_fri_c0_1000_10", X=X, y=y))

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "prokshitha/home-value-insights",
    "house_price_regression_dataset.csv",
)
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="HousePrice", X=X, y=y))

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "fedesoriano/the-boston-houseprice-data",
    "boston.csv",
)
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="BostonHousePrice", X=X, y=y))

df = fetch_data('609_fri_c0_1000_5')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="609_fri_c0_1000_5", X=X, y=y))

df = fetch_data('573_cpu_act')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="573_cpu_act", X=X, y=y))

df = fetch_data('218_house_8L')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="218_house_8L", X=X, y=y))

df = fetch_data('574_house_16H')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="574_house_16H", X=X, y=y))

df = fetch_data('197_cpu_act')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="197_cpu_act", X=X, y=y))

df = fetch_data('227_cpu_small')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="227_cpu_small", X=X, y=y))

df = fetch_data('562_cpu_small')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="562_cpu_small", X=X, y=y))

df = fetch_data('294_satellite_image')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="294_satellite_image", X=X, y=y))

df = fetch_data('666_rmftsa_ladata')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="666_rmftsa_ladata", X=X, y=y))

df = fetch_data('547_no2')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="547_no2", X=X, y=y))

df = fetch_data('623_fri_c4_1000_10')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="623_fri_c4_1000_10", X=X, y=y))

df = fetch_data('537_houses')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="537_houses", X=X, y=y))

df = fetch_data('225_puma8NH')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="225_puma8NH", X=X, y=y))

dataset = fetch_ucirepo(id=477)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
DATASETS.append(AbstractDataset(name="RealEstate", X=X, y=y))

dataset = fetch_ucirepo(id=291)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
DATASETS.append(AbstractDataset(name="Airfoil-Self-Noise", X=X, y=y))

dataset = fetch_ucirepo(id=165)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
DATASETS.append(AbstractDataset(name="ConcreteCompressiveStrength", X=X, y=y))
    
df = fetch_data('529_pollen')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="529_pollen", X=X, y=y))

df = fetch_data('503_wind')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="503_wind", X=X, y=y))
    
df = fetch_data('654_fri_c0_500_10')
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
DATASETS.append(AbstractDataset(name="654_fri_c0_500_10", X=X, y=y))

dataset = fetch_ucirepo(id=186)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
DATASETS.append(AbstractDataset(name="WineQuality", X=X, y=y))

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
DATASETS.append(ParkinsonsTelemonitoringDataset())

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
DATASETS.append(AuctionVerificationDataset())


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