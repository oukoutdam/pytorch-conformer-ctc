from src.pytorch_conformer_ctc.base_config import TrainValidLoopConfig

config = TrainValidLoopConfig(
    train_manifest_path="/home/oukoutdam_slp/datasets/nemo/manifest/train_sp/train_sp_manifest.json",
    valid_manifest_path="/home/oukoutdam_slp/datasets/nemo/manifest/valid/valid_manifest.json"
)
