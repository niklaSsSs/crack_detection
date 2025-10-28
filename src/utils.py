"""
Utility functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


def setup_logging(log_dir, checkpoint_dir, viz_dir):
    """Setup logging directories"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(viz_dir).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, config=None):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(load_path, model=None, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(load_path, map_location='cpu')

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def compute_metrics(labels, preds, probs):
    """Compute classification metrics"""

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
    """Compute per-domain metrics"""

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

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AUC
    axes[0, 1].plot(epochs, history['train_auc'], label='Train', marker='o')
    axes[0, 1].plot(epochs, history['val_auc'], label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title('ROC-AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1
    axes[1, 0].plot(epochs, history['train_f1'], label='Train', marker='o')
    axes[1, 0].plot(epochs, history['val_f1'], label='Val', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], marker='o', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
