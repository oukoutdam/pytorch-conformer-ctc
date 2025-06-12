from src.scripts.configs.example_config import config
from src.pytorch_conformer_ctc.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()
