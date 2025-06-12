# Do not use this config as is
# Create a new config at src/scripts/config

import torch
from dataclasses import dataclass

@dataclass
class TrainValidLoopConfig:
    train_manifest_path : str  # example: "/home/user/datasets/nemo/manifest/train_sp/train_sp_manifest.json"
    valid_manifest_path : str  # example: "/home/user/datasets/nemo/manifest/valid/valid_manifest.json"
    experiment_name : str = "base_experiment"
    checkpoint_dir : str = "checkpoints"
    sample_rate : int = 16000
    max_epochs : int = 4
    batch_size : int = 24
    learning_rate : float = 1e-4
    device : torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
