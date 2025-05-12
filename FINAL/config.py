import uuid
import os
from typing import Optional, List, Union, Dict, Any

class Config:
    """
    Configuration class for counterfactual data augmentation experiments.
    Generates a unique UUID for each experiment run and stores experiment parameters.
    """
    def __init__(
        self,
        # Dataset parameters
        dataset_path: str = "./FINAL/datasets/197_cpu_act.csv",
        dataset_dir__path: str = "./FINAL/datasets",

        # Model parameters
        baseline: str = "mlp",  # Options: "mlp", "xgboost"
        
        # Data generation parameters
        aug_data_size_factor: int = 2.0, # 2.0 means 2x the original data size
        perturb_percent: float = 0.05,  # Percentage to perturb with
        
        # Causal parameters
        alpha: float = 0.05,  # Significance level for causal tests
        indep_test: str = 'fisherz',  # Type of independence test for causal discovery
        
        # Experiment parameters
        test_size: float = 0.2,  # Test split ratio
        random_seed: int = 42,  # Random seed for reproducibility
        num_seeds: int = 5,  # Number of seeds to run as trials
        hyperparam_tune: bool = True,
        
        # Output parameters
        results_dir: str = "./FINAL/results",
        save_plots: bool = True,
        save_models: bool = True,
        
        # Additional parameters that can be passed as a dictionary
        **kwargs: Any
    ):
        # Generate a unique ID for this experiment run
        self.uuid = str(uuid.uuid4())
        self.experiment_name = f"{baseline}_{self.uuid}"
        
        # Dataset parameters
        self.dataset_path = dataset_path
        self.dataset_dir__path = dataset_dir__path
        
        # Model parameters
        self.baseline = baseline
        
        # Data generation parameters
        self.aug_data_size_factor = aug_data_size_factor
        self.perturb_percent = perturb_percent
        
        # Causal parameters
        self.alpha = alpha
        self.indep_test = indep_test
        
        # Experiment parameters
        self.test_size = test_size
        self.random_seed = random_seed
        self.num_seeds = num_seeds
        self.hyperparam_tune = hyperparam_tune

        # Output parameters
        self.results_dir = results_dir
        self.save_plots = save_plots
        self.save_models = save_models
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_experiment_dir(self) -> str:
        """
        Returns the directory path for this experiment's results.
        """
        exp_dir = os.path.join(self.results_dir, self.uuid)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the config to a dictionary for serialization.
        """
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create a Config instance from a dictionary.
        """
        uuid_val = config_dict.pop('uuid', None)
        config = cls(**config_dict)
        if uuid_val:
            config.uuid = uuid_val
        return config
    
    def __str__(self) -> str:
        """
        Returns a string representation of the config.
        """
        return f"Experiment Config (UUID: {self.uuid})\n" + "\n".join(
            f"  {k}: {v}" for k, v in self.__dict__.items() if k != 'uuid'
        )
