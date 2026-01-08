import os
import csv
import json
from pathlib import Path
import torch


class DataLogger:
    def __init__(self, base_dir, activation, experiment_name):
        """
        base_dir: root results directory (e.g., ./results)
        activation: 'relu', 'tanh', etc.
        experiment_name: 'dense', 'prune_0.8', 'rewind_0.01', etc.
        """
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / activation / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.exp_dir / "metrics.csv"
        self.config_path = self.exp_dir / "config.json"
        self.final_path = self.exp_dir / "final.json"

        # Initialize CSV
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "lr",
                    "sparsity"
                ])

    def log_config(self, config_dict):
        clean_config = {}

        for k, v in config_dict.items():
            if isinstance(v, torch.device):
                clean_config[k] = str(v)
            else:
                clean_config[k] = v

        with open(self.config_path, "w") as f:
            json.dump(clean_config, f, indent=4)

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, sparsity=None):
        """Append one epoch of metrics."""
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                lr,
                sparsity if sparsity is not None else ""
            ])

    def log_final(self, final_dict):
        """Save final summary metrics."""
        with open(self.final_path, "w") as f:
            json.dump(final_dict, f, indent=4)

    def save_mask(self, mask_tensor, name="mask.pt"):
        """Save pruning mask or other tensors."""
        torch.save(mask_tensor, self.exp_dir / name)

    def save_model(self, model_state, name="model.pth"):
        """Save model weights."""
        torch.save(model_state, self.exp_dir / name)
