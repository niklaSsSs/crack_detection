"""
Unified Training Script für Mac und Server
Unterstützt beide Konfigurationen via YAML
"""

import os
import sys
import yaml
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Local imports
sys.path.append(str(Path(__file__).parent))
from dataset import SDNET2018Dataset, create_grouped_split, load_dataset
from utils import (
    setup_logging, save_checkpoint, load_checkpoint,
    plot_confusion_matrix, plot_training_history,
    compute_metrics, compute_domain_metrics
)


def set_seed(seed):
    """Set seeds für Reproduzierbarkeit"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(config_path):
    """Load YAML config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_transforms(config, is_train=True):
    """Get augmentations based on config"""

    if not is_train:
        return A.Compose([
            A.Resize(config['data']['input_size'], config['data']['input_size']),
            A.Normalize(mean=config['data']['mean'], std=config['data']['std']),
            ToTensorV2()
        ])

    # Training augmentations
    aug_cfg = config['augmentation']['train']
    transforms = [
        A.Resize(config['data']['input_size'], config['data']['input_size']),
    ]

    # Geometric
    if aug_cfg.get('horizontal_flip', 0) > 0:
        transforms.append(A.HorizontalFlip(p=aug_cfg['horizontal_flip']))

    if aug_cfg.get('vertical_flip', 0) > 0:
        transforms.append(A.VerticalFlip(p=aug_cfg['vertical_flip']))

    if aug_cfg.get('rotate_prob', 0) > 0:
        transforms.append(A.Rotate(
            limit=aug_cfg.get('rotate_limit', 10),
            p=aug_cfg['rotate_prob']
        ))

    if aug_cfg.get('shift_scale_rotate', 0) > 0:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=aug_cfg.get('rotate_limit', 10),
            p=aug_cfg['shift_scale_rotate']
        ))

    if aug_cfg.get('perspective', 0) > 0:
        transforms.append(A.Perspective(p=aug_cfg['perspective']))

    # Photometric
    if aug_cfg.get('brightness_contrast', 0) > 0:
        transforms.append(A.RandomBrightnessContrast(p=aug_cfg['brightness_contrast']))

    if aug_cfg.get('hue_saturation', 0) > 0:
        transforms.append(A.HueSaturationValue(p=aug_cfg['hue_saturation']))

    if aug_cfg.get('random_gamma', 0) > 0:
        transforms.append(A.RandomGamma(p=aug_cfg['random_gamma']))

    if aug_cfg.get('gauss_noise', 0) > 0:
        transforms.append(A.GaussNoise(p=aug_cfg['gauss_noise']))

    if aug_cfg.get('gauss_blur', 0) > 0:
        transforms.append(A.GaussianBlur(p=aug_cfg['gauss_blur']))

    if aug_cfg.get('motion_blur', 0) > 0:
        transforms.append(A.MotionBlur(p=aug_cfg['motion_blur']))

    if aug_cfg.get('image_compression', 0) > 0:
        transforms.append(A.ImageCompression(
            quality_lower=75,
            quality_upper=100,
            p=aug_cfg['image_compression']
        ))

    # Advanced (nur Server)
    if aug_cfg.get('grid_distortion', 0) > 0:
        transforms.append(A.GridDistortion(p=aug_cfg['grid_distortion']))

    if aug_cfg.get('optical_distortion', 0) > 0:
        transforms.append(A.OpticalDistortion(p=aug_cfg['optical_distortion']))

    # Normalize and convert
    transforms.extend([
        A.Normalize(mean=config['data']['mean'], std=config['data']['std']),
        ToTensorV2()
    ])

    return A.Compose(transforms)


class CrackClassifier(nn.Module):
    """EfficientNet-B0 Classifier"""

    def __init__(self, config):
        super().__init__()

        if config['model']['pretrained']:
            # Versuche lokale Weights zu laden (für offline Server)
            local_weights_path = Path(__file__).parent.parent / "pretrained_weights" / "efficientnet_b0_rwightman-7f5810bc.pth"

            if local_weights_path.exists():
                print(f"  ✓ Loading weights from: {local_weights_path.name}")
                self.encoder = efficientnet_b0(weights=None)
                state_dict = torch.load(local_weights_path, map_location='cpu')
                self.encoder.load_state_dict(state_dict)
            else:
                # Fallback: Download von PyTorch (braucht Internet)
                print(f"  ⚠ Local weights not found, downloading from PyTorch...")
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                self.encoder = efficientnet_b0(weights=weights)
        else:
            self.encoder = efficientnet_b0(weights=None)

        in_features = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config['model'].get('dropout', 0.2)),
            nn.Linear(in_features, config['model']['num_classes'])
        )

    def forward(self, x):
        features = self.encoder(x)
        if len(features.shape) == 4:
            features = features.flatten(1)
        logits = self.classifier(features.unsqueeze(-1).unsqueeze(-1))
        return logits.squeeze(-1)

    def get_encoder(self):
        return self.encoder


class Trainer:
    """Unified Trainer für Mac und Server"""

    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path

        # Setup
        set_seed(config['experiment']['seed'])
        self.setup_device()
        self.setup_logging()

        # Data
        self.setup_data()

        # Model
        self.setup_model()

        # Training
        self.setup_training()

        # State
        self.current_epoch = 0
        self.best_val_auc = 0
        self.best_epoch = 0
        self.history = {
            'train_loss': [], 'train_auc': [], 'train_f1': [],
            'val_loss': [], 'val_auc': [], 'val_f1': [],
            'lr': []
        }

    def setup_device(self):
        """Setup device"""
        device_str = self.config['experiment']['device']

        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            if self.config.get('hardware', {}).get('gpu_id') is not None:
                torch.cuda.set_device(self.config['hardware']['gpu_id'])
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif device_str == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using MPS (Apple Silicon)")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Threading für CPU
        if self.device.type == 'cpu' and 'hardware' in self.config:
            num_threads = self.config['hardware'].get('num_threads', 4)
            torch.set_num_threads(num_threads)
            print(f"CPU threads: {num_threads}")

        # Deterministic
        if self.config.get('hardware', {}).get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    def setup_logging(self):
        """Setup logging directories"""
        log_cfg = self.config['logging']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config['experiment']['name']}_{timestamp}"

        self.log_dir = Path(log_cfg['log_dir']) / exp_name
        self.checkpoint_dir = Path(log_cfg['checkpoint_dir']) / exp_name
        self.viz_dir = Path(log_cfg['visualization_dir']) / exp_name

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_save_path = self.log_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print(f"\nExperiment: {exp_name}")
        print(f"Logs: {self.log_dir}")

    def setup_data(self):
        """Setup datasets and dataloaders"""

        data_cfg = self.config['data']

        # Load dataset
        file_paths, labels, domains, series_ids = load_dataset(data_cfg['data_dir'])

        # Create splits
        splits = create_grouped_split(file_paths, labels, domains, series_ids)

        # Create datasets
        train_transform = get_transforms(self.config, is_train=True)
        val_transform = get_transforms(self.config, is_train=False)

        self.train_dataset = SDNET2018Dataset(
            splits['train']['paths'],
            splits['train']['labels'],
            splits['train']['domains'],
            transform=train_transform
        )

        self.val_dataset = SDNET2018Dataset(
            splits['val']['paths'],
            splits['val']['labels'],
            splits['val']['domains'],
            transform=val_transform
        )

        self.test_dataset = SDNET2018Dataset(
            splits['test']['paths'],
            splits['test']['labels'],
            splits['test']['domains'],
            transform=val_transform
        )

        # Weighted sampler
        train_labels = splits['train']['labels']
        class_counts = np.bincount(train_labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels.astype(int)]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Dataloaders
        train_cfg = self.config['training']
        pin_memory = data_cfg.get('pin_memory', False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg['batch_size'],
            sampler=train_sampler,
            num_workers=data_cfg['num_workers'],
            pin_memory=pin_memory,
            persistent_workers=self.config.get('performance', {}).get('persistent_workers', False)
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=pin_memory,
            persistent_workers=self.config.get('performance', {}).get('persistent_workers', False)
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=pin_memory
        )

        print(f"\nDatasets:")
        print(f"  Train: {len(self.train_dataset)}")
        print(f"  Val: {len(self.val_dataset)}")
        print(f"  Test: {len(self.test_dataset)}")

        # Store splits for later
        self.splits = splits

    def setup_model(self):
        """Setup model"""

        self.model = CrackClassifier(self.config).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nModel: {self.config['model']['backbone']}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    def setup_training(self):
        """Setup optimizer, scheduler, loss"""

        train_cfg = self.config['training']

        # Loss
        loss_cfg = train_cfg['loss']

        if loss_cfg.get('pos_weight_auto', False):
            n_crack = (self.splits['train']['labels'] == 1).sum()
            n_noncrack = (self.splits['train']['labels'] == 0).sum()
            pos_weight = torch.tensor([n_noncrack / n_crack], dtype=torch.float32).to(self.device)
        else:
            pos_weight = None

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if pos_weight is not None:
            print(f"\nLoss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.2f})")
        else:
            print(f"\nLoss: BCEWithLogitsLoss")

        # Optimizer
        opt_cfg = train_cfg['optimizer']

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=opt_cfg.get('betas', [0.9, 0.999])
        )

        # Scheduler
        sched_cfg = train_cfg['scheduler']
        warmup_epochs = sched_cfg.get('warmup_epochs', 1)
        total_epochs = train_cfg['epochs']

        def warmup_cosine_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_cosine_schedule)

        # Mixed precision
        self.use_amp = self.config['experiment'].get('mixed_precision', False) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using mixed precision (FP16)")

        # Gradient accumulation
        self.grad_accum_steps = self.config.get('performance', {}).get('gradient_accumulation_steps', 1)
        if self.grad_accum_steps > 1:
            print(f"Gradient accumulation: {self.grad_accum_steps} steps")

        # Early stopping
        es_cfg = train_cfg['early_stopping']
        self.patience = es_cfg['patience']
        self.min_delta = es_cfg['min_delta']
        self.patience_counter = 0

    def train_epoch(self):
        """Train one epoch"""

        self.model.train()

        total_loss = 0
        all_labels = []
        all_probs = []
        all_preds = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.grad_accum_steps
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps

            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.config['training'].get('grad_clip'):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Metrics
            total_loss += loss.item() * self.grad_accum_steps
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)

            pbar.set_postfix({'loss': f'{loss.item() * self.grad_accum_steps:.4f}'})

        # Compute metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(self.train_loader)

        return metrics

    @torch.no_grad()
    def validate(self):
        """Validate"""

        self.model.eval()

        total_loss = 0
        all_labels = []
        all_probs = []
        all_preds = []
        all_domains = []

        pbar = tqdm(self.val_loader, desc="Validating")

        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            domains = batch['domain']

            # Forward
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_domains.extend(domains)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_domains = np.array(all_domains)

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(self.val_loader)

        # Domain metrics
        if self.config['logging'].get('domain_metrics', False):
            domain_metrics = compute_domain_metrics(all_labels, all_preds, all_probs, all_domains)
            metrics['domain_metrics'] = domain_metrics

        return metrics

    def train(self):
        """Main training loop"""

        print("\n" + "=" * 80)
        print("TRAINING START")
        print("=" * 80)

        epochs = self.config['training']['epochs']

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 80)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {lr:.6f}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Scheduler step
            self.scheduler.step()

            # Log
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Domain metrics
            if 'domain_metrics' in val_metrics:
                print("\nDomain Metrics:")
                for domain, dm in val_metrics['domain_metrics'].items():
                    auc_str = f"{dm['auc']:.4f}" if dm['auc'] is not None else "N/A"
                    print(f"  {domain:10} - AUC: {auc_str}, F1: {dm['f1']:.4f}")

            # History
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(lr)

            # Save best
            if val_metrics['auc'] > self.best_val_auc + self.min_delta:
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch
                self.patience_counter = 0

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics,
                    self.checkpoint_dir / 'best_model.pt',
                    config=self.config
                )

                print(f"\n✓ Best model saved (AUC: {self.best_val_auc:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")

        # Test
        self.test()

        # Export encoder
        self.export_encoder()

    @torch.no_grad()
    def test(self):
        """Test on test set"""

        print("\n" + "=" * 80)
        print("TEST EVALUATION")
        print("=" * 80)

        # Load best model
        checkpoint = load_checkpoint(
            self.checkpoint_dir / 'best_model.pt',
            self.model,
            self.optimizer,
            self.scheduler
        )

        self.model.eval()

        total_loss = 0
        all_labels = []
        all_probs = []
        all_preds = []
        all_domains = []

        pbar = tqdm(self.test_loader, desc="Testing")

        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            domains = batch['domain']

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_domains.extend(domains)

        # Metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_domains = np.array(all_domains)

        test_metrics = compute_metrics(all_labels, all_preds, all_probs)
        test_metrics['loss'] = total_loss / len(self.test_loader)

        domain_metrics = compute_domain_metrics(all_labels, all_preds, all_probs, all_domains)
        test_metrics['domain_metrics'] = domain_metrics

        print(f"\nTest - Loss: {test_metrics['loss']:.4f}")
        print(f"       AUC: {test_metrics['auc']:.4f}")
        print(f"       F1: {test_metrics['f1']:.4f}")
        print(f"       Precision: {test_metrics['precision']:.4f}")
        print(f"       Recall: {test_metrics['recall']:.4f}")

        print("\nDomain Metrics:")
        for domain, dm in domain_metrics.items():
            auc_str = f"{dm['auc']:.4f}" if dm['auc'] is not None else "N/A"
            print(f"  {domain:10} - AUC: {auc_str}, F1: {dm['f1']:.4f}")

        # Save metrics
        import json
        test_metrics_clean = {
            'loss': float(test_metrics['loss']),
            'auc': float(test_metrics['auc']),
            'f1': float(test_metrics['f1']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'domain_metrics': {
                domain: {
                    'auc': float(m['auc']) if m['auc'] is not None else None,
                    'f1': float(m['f1']),
                    'n_samples': int(m['n_samples'])
                }
                for domain, m in domain_metrics.items()
            }
        }

        with open(self.log_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics_clean, f, indent=2)

        # Plots
        plot_confusion_matrix(test_metrics['confusion_matrix'], self.viz_dir / 'confusion_matrix.png')
        plot_training_history(self.history, self.viz_dir / 'training_history.png')

        print(f"\n✓ Test results saved to {self.log_dir}")

        return test_metrics

    def export_encoder(self):
        """Export encoder weights"""

        encoder_path = self.checkpoint_dir / 'encoder_weights.pt'

        encoder_state = {
            'encoder_state_dict': self.model.get_encoder().state_dict(),
            'config': {
                'backbone': self.config['model']['backbone'],
                'input_size': self.config['data']['input_size'],
                'mean': self.config['data']['mean'],
                'std': self.config['data']['std'],
                'pretrained_on': 'SDNET2018',
                'task': 'crack_detection',
                'best_val_auc': float(self.best_val_auc)
            }
        }

        torch.save(encoder_state, encoder_path)
        print(f"\n✓ Encoder weights exported: {encoder_path}")


def main():
    parser = argparse.ArgumentParser(description="Train crack encoder")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/config_mac.yaml)'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    trainer = Trainer(config, args.config)
    trainer.train()


if __name__ == "__main__":
    main()
