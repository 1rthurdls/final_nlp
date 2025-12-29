"""
Training module for ABSA models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Dict, Optional
import yaml

from models.advanced import BertForABSA, BertForABSAWithCRF
from models.baseline import BiLSTMABSA
from src.preprocessing import create_dataloaders, SemEvalDataLoader
from src.evaluation import compute_metrics, evaluate_model


class ABSATrainer:
    """Trainer for ABSA models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['model']['learning_rate']
        self.warmup_steps = config['model']['warmup_steps']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.save_dir = config['training']['save_dir']

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }

    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.num_epochs
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                aspect_labels=batch['aspect_labels'],
                sentiment_labels=batch['sentiment_labels']
            )

            loss = outputs['loss']

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0

        all_aspect_preds = []
        all_aspect_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    aspect_labels=batch['aspect_labels'],
                    sentiment_labels=batch['sentiment_labels']
                )

                if outputs['loss'] is not None:
                    total_loss += outputs['loss'].item()

                # Get predictions
                aspect_logits = outputs['aspect_logits']
                sentiment_logits = outputs['sentiment_logits']

                aspect_preds = torch.argmax(aspect_logits, dim=-1)
                sentiment_preds = torch.argmax(sentiment_logits, dim=-1)

                # Collect predictions and labels
                for i in range(batch['input_ids'].size(0)):
                    mask = batch['attention_mask'][i].bool()

                    # Aspect predictions
                    aspect_pred = aspect_preds[i][mask].cpu().numpy()
                    aspect_label = batch['aspect_labels'][i][mask].cpu().numpy()
                    all_aspect_preds.append(aspect_pred)
                    all_aspect_labels.append(aspect_label)

                    # Sentiment predictions (only for aspect tokens)
                    sentiment_pred = sentiment_preds[i][mask].cpu().numpy()
                    sentiment_label = batch['sentiment_labels'][i][mask].cpu().numpy()

                    # Filter out -100 (ignore) labels
                    valid_idx = sentiment_label != -100
                    if valid_idx.any():
                        all_sentiment_preds.append(sentiment_pred[valid_idx])
                        all_sentiment_labels.append(sentiment_label[valid_idx])

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)

        # Aspect extraction metrics
        aspect_metrics = compute_metrics(all_aspect_labels, all_aspect_preds)

        # Sentiment classification metrics
        sentiment_metrics = {}
        if all_sentiment_labels:
            sentiment_preds_flat = np.concatenate(all_sentiment_preds)
            sentiment_labels_flat = np.concatenate(all_sentiment_labels)
            sentiment_metrics = compute_metrics(
                [sentiment_labels_flat],
                [sentiment_preds_flat],
                prefix="sentiment_"
            )

        # Combine metrics
        metrics = {
            'val_loss': avg_loss,
            **aspect_metrics,
            **sentiment_metrics
        }

        return metrics

    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self.train_epoch()
            print(f"Train loss: {train_loss:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"Val loss: {val_metrics['val_loss']:.4f}")
            print(f"Val F1: {val_metrics.get('f1', 0.0):.4f}")
            print(f"Val Precision: {val_metrics.get('precision', 0.0):.4f}")
            print(f"Val Recall: {val_metrics.get('recall', 0.0):.4f}")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_f1'].append(val_metrics.get('f1', 0.0))
            self.history['val_precision'].append(val_metrics.get('precision', 0.0))
            self.history['val_recall'].append(val_metrics.get('recall', 0.0))

            # Early stopping and model saving
            current_f1 = val_metrics.get('f1', 0.0)
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"New best model saved with F1: {self.best_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save checkpoint every epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

        print(f"\nTraining completed. Best F1: {self.best_f1:.4f}")

        # Save training history
        self.save_history()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'epoch': len(self.history['train_loss']),
            'config': self.config
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def save_history(self):
        """Save training history."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_f1 = checkpoint['best_f1']

        print(f"Checkpoint loaded from {path}")
        print(f"Best F1: {self.best_f1:.4f}")


def train_model(config_path: str = "configs/config.yaml"):
    """
    Main training function.

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    # Device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_path=os.path.join(config['data']['raw_dir'], config['data']['train_file']),
        test_path=os.path.join(config['data']['raw_dir'], config['data']['test_file']),
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_seq_length']
    )

    # Initialize model
    from transformers import AutoConfig

    bert_config = AutoConfig.from_pretrained(config['model']['name'])
    model = BertForABSA(
        bert_config,
        num_aspect_labels=config['model']['num_aspect_labels'],
        num_sentiment_labels=config['model']['num_aspect_labels']
    )

    # Initialize trainer
    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
        device=device
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    train_model()
