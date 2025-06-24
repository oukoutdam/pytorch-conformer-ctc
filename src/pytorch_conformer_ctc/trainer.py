import os
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import wandb
from jiwer import cer
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from src.pytorch_conformer_ctc.tokens import TOKENS
from src.pytorch_conformer_ctc.data.transform import TextTransform
from src.pytorch_conformer_ctc.data.datasets import CEJCDataset, collate_fn
from src.pytorch_conformer_ctc.models import PytorchConformerCTC
from src.pytorch_conformer_ctc.modules.specaugment import SpecAugment
from src.pytorch_conformer_ctc.utils.decoding import greedy_ctc_decode


class IterMeter:
    def __init__(self) -> None:
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


class Trainer:
    def __init__(self, config):
        """
        Initialize the Trainer with configuration.

        Args:
            config: TrainValidLoopConfig dataclass instance
        """
        self.config = config
        self.device = config.device

        # Initialize tokenizer
        self.tokenizer = TextTransform(TOKENS)

        # Initialize transforms
        self._setup_transforms()

        # Initialize datasets and dataloaders
        self._setup_data()

        # Initialize model
        self._setup_model()

        # Initialize optimizer and criterion
        self._setup_training_components()

        # Initialize tracking
        self.itermeter = IterMeter()

        # Setup checkpoint directory
        self._setup_checkpoints()

        # Initialize wandb
        self._setup_wandb()

    def _setup_transforms(self):
        """Setup audio transforms for training and validation."""
        # input: [num_channels, samples]
        # output after MelSpectrogram: [num_channels, n_mels, time_frames]
        # output after amplitudetodb: [num_channels, n_mels, time_frames] but scaled to db scale
        # self.audio_to_logmel_transform = torch.nn.Sequential(
        #     MelSpectrogram(
        #         sample_rate=self.config.sample_rate,
        #         n_fft=512,
        #         win_length=int(0.025 * self.config.sample_rate),  # 400 for 16kHz
        #         hop_length=int(0.01 * self.config.sample_rate),   # 160 for 16kHz
        #         power=2,
        #         n_mels=80,
        #     ),
        #     AmplitudeToDB(
        #         stype="power",
        #         top_db=80,
        #     ),
        # )

        # self.train_transform = torch.nn.Sequential(
        #     self.audio_to_logmel_transform,
        #     SpecAugment(
        #         freq_masks=2,
        #         freq_width=27,
        #         time_masks=10,
        #         time_width_ratio=0.05
        #     )
        # )

        # self.test_transform = self.audio_to_logmel_transform

        # moving transforms into preprocessor
        self.train_transform = None
        self.test_transform = None

    def _setup_data(self):
        """Setup datasets and dataloaders."""
        self.train_dataset = CEJCDataset(
            manifest_path=self.config.train_manifest_path,
            tokenizer=self.tokenizer,
            transform=self.train_transform,
        )

        self.valid_dataset = CEJCDataset(
            manifest_path=self.config.valid_manifest_path,
            tokenizer=self.tokenizer,
            transform=self.test_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Valid dataset size: {len(self.valid_dataset)}")

    def _setup_model(self):
        """Setup the model."""
        self.model = PytorchConformerCTC(
            input_dim=80,
            num_heads=8,
            d_model=256,
            num_layers=16,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
            subsampling_factor=4,
            ff_expansion_factor=4,
            vocab_size=len(self.tokenizer),
        ).to(self.device)

        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    def _setup_training_components(self):
        """Setup optimizer and loss criterion."""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = CTCLoss(blank=self.tokenizer.blank, zero_infinity=True)

    def _setup_checkpoints(self):
        """Setup checkpoint directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"{self.config.experiment_name}_{timestamp}"
        self.run_directory = os.path.join(self.config.checkpoint_dir, experiment_dir)
        os.makedirs(self.run_directory, exist_ok=True)
        print(f"Checkpoints will be saved to: {self.run_directory}")

    def _setup_wandb(self):
        """Initialize wandb logging."""
        self.run = wandb.init(
            project="diy-conformer-ctc",
            name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.max_epochs,
                "batch_size": self.config.batch_size,
                "sample_rate": self.config.sample_rate,
                "experiment_name": self.config.experiment_name,
            }
        )

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()

        for batch_idx, _data in enumerate(self.train_loader):
            feats, flens, labs, llens = _data
            feats, labs = feats.to(self.device), labs.to(self.device)
            flens, llens = flens.to(self.device), llens.to(self.device)

            self.optimizer.zero_grad()
            logp, out_lens = self.model(feats, flens)
            logp = logp.transpose(0, 1)  # [B, T, Vocab] -> [T, B, Vocab]
            loss = self.criterion(logp, labs, out_lens, llens)
            loss.backward()
            self.optimizer.step()
            self.itermeter.step()

            wandb.log({"train_loss": loss.item(), "iteration": self.itermeter.get()})

            if batch_idx % 100 == 0 or batch_idx == len(self.train_loader) - 1:
                total_batches = len(self.train_loader)
                progress_pct = 100.0 * (batch_idx + 1) / total_batches
                print(
                    f"Train Epoch: [Batch {batch_idx + 1}/{total_batches} "
                    f"({progress_pct:.0f}%)]\tLoss: {loss.item():.6f}"
                )

    def validate(self) -> tuple[float, list[str], list[str]]:
        """Run validation and return loss, predictions, and targets."""
        total_loss = 0.0
        all_preds = []
        all_targets = []
        self.model.eval()

        with torch.no_grad():
            for feats, flens, llabs, llens in self.valid_loader:
                feats, flens = feats.to(self.device), flens.to(self.device)
                llabs, llens = llabs.to(self.device), llens.to(self.device)

                # Forward pass
                logp, out_lens = self.model(feats, flens)

                # Calculate loss
                loss = self.criterion(logp.transpose(0, 1), llabs, out_lens, llens)
                total_loss += loss.item()

                # Decode predictions
                decoded_seqs = greedy_ctc_decode(logp, out_lens, blank_idx=self.tokenizer.blank)
                batch_preds = [self.tokenizer.int_to_text(seq) for seq in decoded_seqs]
                all_preds.extend(batch_preds)

                # Decode targets
                targets = llabs.cpu().tolist()
                for target_seq, tgt_len in zip(targets, llens.cpu().tolist()):
                    all_targets.append(self.tokenizer.int_to_text(target_seq[:tgt_len]))

        avg_loss = total_loss / len(self.valid_loader)
        return avg_loss, all_preds, all_targets

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        ckpt_path = os.path.join(self.run_directory, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from: {checkpoint_path}")
        return checkpoint.get('epoch', 0)

    def log_predictions(self, epoch: int, preds: list[str], targets: list[str]):
        """Log prediction examples to wandb."""
        examples = []
        for i in range(min(5, len(preds))):
            examples.append([targets[i], len(targets[i]), preds[i], len(preds[i])])

        wandb.log({
            f"predictions_epoch_{epoch}": wandb.Table(
                columns=["target", "target_len", "prediction", "prediction_len"],
                data=examples
            )
        })

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Experiment: {self.config.experiment_name}")

        for epoch in range(self.config.max_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.max_epochs} ---")

            # Training
            self.train_epoch()
            print(f"Epoch {epoch} training finished")

            # Validation
            val_loss, preds, targets = self.validate()
            print(f"Epoch {epoch} validation loss: {val_loss:.6f}")

            # Calculate metrics
            character_error_rate = cer(targets, preds)
            print(f"Epoch {epoch} CER: {character_error_rate:.4f}")

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "cer": character_error_rate,
            })

            # Log prediction examples
            self.log_predictions(epoch, preds, targets)

            # Save checkpoint
            self.save_checkpoint(epoch)

        print("\nTraining completed!")
        wandb.finish()
