"""
SDNET2018 Encoder Pre-training - CPU Optimized Version
Optimized for CPU training with reduced input size and batch size
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"  # Optimize CPU threading

import json
import yaml
import random
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration - Optimized for CPU
CONFIG = {
    'data_dir': '/Users/niklaswegner/python_proj/crack_model/SDNET2018',
    'output_dir': '/Users/niklaswegner/python_proj/crack_model/runs',
    'experiment_name': f'efficientnet_b0_crack_pretraining_cpu_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model
    'backbone': 'efficientnet_b0',
    'pretrained': True,
    'input_size': 256,  # Keep original size (faster)

    # Training - Optimized for faster iteration
    'batch_size': 64,  # Larger batch for CPU
    'epochs': 20,  # Reduced from 30
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'warmup_epochs': 1,  # Reduced from 2

    # Data
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'num_workers': 4,

    # Early stopping
    'patience': 5,
    'min_delta': 0.001,

    # Normalization (ImageNet)
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],

    # Class mapping
    'domain_map': {'D': 'Bridge', 'P': 'Pavement', 'W': 'Wall'},
    'class_map': {
        'CD': 1, 'UD': 0,  # Bridge
        'CP': 1, 'UP': 0,  # Pavement
        'CW': 1, 'UW': 0   # Wall
    }
}

# Force CPU
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")
print(f"CPU threads: {torch.get_num_threads()}")


class SDNET2018Dataset(Dataset):
    """SDNET2018 Dataset with domain-aware loading"""

    def __init__(self, file_paths, labels, domains, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.domains = domains
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        domain = self.domains[idx]

        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'domain': domain,
            'path': str(img_path)
        }


def get_train_transforms(input_size=256):
    """Training augmentations - Simplified"""
    return A.Compose([
        A.Resize(input_size, input_size),

        # Photometric augmentations
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=7, p=0.3),

        # Normalize and convert to tensor
        A.Normalize(mean=CONFIG['mean'], std=CONFIG['std']),
        ToTensorV2()
    ])


def get_val_transforms(input_size=256):
    """Validation/Test augmentations"""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=CONFIG['mean'], std=CONFIG['std']),
        ToTensorV2()
    ])


def extract_series_id(file_path):
    """Extract series ID from filename"""
    stem = Path(file_path).stem
    parts = stem.split('-')
    if len(parts) >= 2:
        return parts[0]
    return stem


def load_dataset():
    """Load dataset"""

    data_dir = Path(CONFIG['data_dir'])

    file_paths = []
    labels = []
    domains = []
    series_ids = []

    print("\nLoading dataset...")

    for domain_code in ['D', 'P', 'W']:
        domain_path = data_dir / domain_code
        domain_name = CONFIG['domain_map'][domain_code]

        if not domain_path.exists():
            continue

        for class_dir in domain_path.iterdir():
            if not class_dir.is_dir():
                continue

            class_code = class_dir.name
            label = CONFIG['class_map'].get(class_code)

            if label is None:
                continue

            for img_file in class_dir.glob("*.jpg"):
                file_paths.append(str(img_file))
                labels.append(label)
                domains.append(domain_name)
                series_ids.append(extract_series_id(img_file))

    file_paths = np.array(file_paths)
    labels = np.array(labels)
    domains = np.array(domains)
    series_ids = np.array(series_ids)

    print(f"Total images: {len(file_paths)}")
    print(f"Crack: {(labels == 1).sum()}, NonCrack: {(labels == 0).sum()}")

    return file_paths, labels, domains, series_ids


def create_grouped_split(file_paths, labels, domains, series_ids):
    """Create grouped split"""

    print("\nCreating grouped split...")

    unique_series = np.unique(series_ids)
    series_to_idx = {s: i for i, s in enumerate(unique_series)}
    group_ids = np.array([series_to_idx[s] for s in series_ids])

    n_splits = 10
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    splits = list(skf.split(file_paths, labels, groups=group_ids))
    train_idx, temp_idx = splits[0]

    temp_labels = labels[temp_idx]
    temp_groups = group_ids[temp_idx]

    skf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
    temp_splits = list(skf2.split(temp_idx, temp_labels, groups=temp_groups))
    val_rel_idx, test_rel_idx = temp_splits[0]

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    splits = {
        'train': {
            'paths': file_paths[train_idx],
            'labels': labels[train_idx],
            'domains': domains[train_idx]
        },
        'val': {
            'paths': file_paths[val_idx],
            'labels': labels[val_idx],
            'domains': domains[val_idx]
        },
        'test': {
            'paths': file_paths[test_idx],
            'labels': labels[test_idx],
            'domains': domains[test_idx]
        }
    }

    for split_name, split_data in splits.items():
        n_total = len(split_data['labels'])
        n_crack = (split_data['labels'] == 1).sum()
        print(f"{split_name.upper()}: {n_total} ({n_crack} crack)")

    return splits


class CrackClassifier(nn.Module):
    """EfficientNet-B0 binary classifier"""

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.encoder = efficientnet_b0(weights=weights)
        else:
            self.encoder = efficientnet_b0(weights=None)

        in_features = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        if len(features.shape) == 4:
            features = features.flatten(1)
        logits = self.classifier(features.unsqueeze(-1).unsqueeze(-1))
        return logits.squeeze(-1)

    def get_encoder(self):
        return self.encoder


def get_weighted_sampler(labels):
    """Create weighted sampler"""
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Warmup + cosine scheduler"""

    def warmup_cosine_schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)


class EarlyStopping:
    """Early stopping"""

    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def compute_metrics(labels, preds, probs):
    """Compute metrics"""

    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }


def compute_domain_metrics(labels, preds, probs, domains):
    """Per-domain metrics"""

    domain_metrics = {}

    for domain in np.unique(domains):
        mask = domains == domain
        if mask.sum() == 0:
            continue

        domain_labels = labels[mask]
        domain_preds = preds[mask]
        domain_probs = probs[mask]

        if len(np.unique(domain_labels)) < 2:
            domain_metrics[domain] = {
                'auc': None,
                'f1': f1_score(domain_labels, domain_preds, zero_division=0),
                'n_samples': mask.sum()
            }
        else:
            domain_metrics[domain] = {
                'auc': roc_auc_score(domain_labels, domain_probs),
                'f1': f1_score(domain_labels, domain_preds, zero_division=0),
                'n_samples': mask.sum()
            }

    return domain_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""

    model.train()

    total_loss = 0
    all_labels = []
    all_probs = []
    all_preds = []

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""

    model.eval()

    total_loss = 0
    all_labels = []
    all_probs = []
    all_preds = []
    all_domains = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        domains = batch['domain']

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_domains.extend(domains)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_domains = np.array(all_domains)

    metrics = compute_metrics(all_labels, all_probs, all_preds)
    metrics['loss'] = total_loss / len(dataloader)

    domain_metrics = compute_domain_metrics(all_labels, all_preds, all_probs, all_domains)
    metrics['domain_metrics'] = domain_metrics

    return metrics


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['NonCrack', 'Crack'],
        yticklabels=['NonCrack', 'Crack']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path):
    """Plot training curves"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_auc'], label='Train', marker='o')
    axes[0, 1].plot(epochs, history['val_auc'], label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title('ROC-AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['train_f1'], label='Train', marker='o')
    axes[1, 0].plot(epochs, history['val_f1'], label='Val', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], marker='o', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Main training pipeline"""

    output_dir = Path(CONFIG['output_dir']) / CONFIG['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SDNET2018 ENCODER PRE-TRAINING (CPU OPTIMIZED)")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

    file_paths, labels, domains, series_ids = load_dataset()
    splits = create_grouped_split(file_paths, labels, domains, series_ids)

    train_dataset = SDNET2018Dataset(
        splits['train']['paths'],
        splits['train']['labels'],
        splits['train']['domains'],
        transform=get_train_transforms(CONFIG['input_size'])
    )

    val_dataset = SDNET2018Dataset(
        splits['val']['paths'],
        splits['val']['labels'],
        splits['val']['domains'],
        transform=get_val_transforms(CONFIG['input_size'])
    )

    test_dataset = SDNET2018Dataset(
        splits['test']['paths'],
        splits['test']['labels'],
        splits['test']['domains'],
        transform=get_val_transforms(CONFIG['input_size'])
    )

    train_sampler = get_weighted_sampler(splits['train']['labels'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=False
    )

    print(f"\nDataloaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    model = CrackClassifier(pretrained=CONFIG['pretrained']).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    n_crack = (splits['train']['labels'] == 1).sum()
    n_noncrack = (splits['train']['labels'] == 0).sum()
    pos_weight = torch.tensor([n_noncrack / n_crack], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Loss: BCEWithLogitsLoss with pos_weight={pos_weight.item():.2f}")

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    scheduler = get_warmup_scheduler(
        optimizer,
        warmup_epochs=CONFIG['warmup_epochs'],
        total_epochs=CONFIG['epochs']
    )

    early_stopping = EarlyStopping(
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        mode='max'
    )

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    history = {
        'train_loss': [], 'train_auc': [], 'train_f1': [],
        'val_loss': [], 'val_auc': [], 'val_f1': [],
        'lr': []
    }

    best_val_auc = 0
    best_epoch = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 80)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")

        history['train_loss'].append(train_metrics['loss'])
        history['train_auc'].append(train_metrics['auc'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_f1': val_metrics['f1'],
                'config': CONFIG
            }

            checkpoint_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"\n✓ Best model saved (AUC: {best_val_auc:.4f})")

        if early_stopping(val_metrics['auc']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\nTest - Loss: {test_metrics['loss']:.4f}")
    print(f"       AUC: {test_metrics['auc']:.4f}")
    print(f"       F1: {test_metrics['f1']:.4f}")
    print(f"       Precision: {test_metrics['precision']:.4f}")
    print(f"       Recall: {test_metrics['recall']:.4f}")

    print("\nTest Domain Metrics:")
    for domain, metrics in test_metrics['domain_metrics'].items():
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
        print(f"  {domain:10} - AUC: {auc_str}, F1: {metrics['f1']:.4f}")

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
            for domain, m in test_metrics['domain_metrics'].items()
        }
    }

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics_clean, f, indent=2)

    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        output_dir / 'confusion_matrix.png'
    )

    plot_training_history(history, output_dir / 'training_history.png')

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    encoder_path = output_dir / 'encoder_sdnet_efficientnet_b0.pt'
    encoder_state = {
        'encoder_state_dict': model.get_encoder().state_dict(),
        'config': {
            'backbone': CONFIG['backbone'],
            'input_size': CONFIG['input_size'],
            'mean': CONFIG['mean'],
            'std': CONFIG['std'],
            'pretrained_on': 'SDNET2018',
            'task': 'crack_detection',
            'val_auc': float(best_val_auc),
            'test_auc': float(test_metrics['auc'])
        }
    }

    torch.save(encoder_state, encoder_path)
    print(f"\n✓ Encoder weights exported: {encoder_path}")

    usage_note = f"""
# Encoder Weights - CPU Training

Backbone: {CONFIG['backbone']}
Input size: {CONFIG['input_size']}x{CONFIG['input_size']}
Val AUC: {best_val_auc:.4f}
Test AUC: {test_metrics['auc']:.4f}

## Usage for Weld Segmentation

```python
import torch
from torchvision.models import efficientnet_b0

checkpoint = torch.load('encoder_sdnet_efficientnet_b0.pt')
encoder = efficientnet_b0(weights=None)
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Use with U-Net/DeepLabv3+ for segmentation
# Freeze encoder for 3-5 epochs, then fine-tune at lr=1e-4
```
"""

    with open(output_dir / 'USAGE.md', 'w') as f:
        f.write(usage_note)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Val AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Target (≥0.95): {'✓ PASS' if test_metrics['auc'] >= 0.95 else '✗ FAIL'}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
