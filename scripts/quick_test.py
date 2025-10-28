#!/usr/bin/env python3
"""
Quick Test - Prüft ob Training startet (1 Batch)
Zum Testen ohne lange zu warten
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from train import Trainer

print("=" * 70)
print("🧪 QUICK TEST")
print("=" * 70)
print()

# Mini Config für Test
config = {
    'experiment': {
        'name': 'test',
        'seed': 42,
        'device': 'cpu',  # Immer CPU für Stabilität
        'mixed_precision': False
    },
    'data': {
        'data_dir': './SDNET2018',
        'input_size': 256,
        'num_workers': 2,  # Weniger für Test
        'pin_memory': False,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'model': {
        'backbone': 'efficientnet_b0',
        'pretrained': True,
        'num_classes': 1,
        'dropout': 0.2
    },
    'training': {
        'batch_size': 8,  # Klein für Test
        'epochs': 1,  # Nur 1 Epoche
        'optimizer': {
            'name': 'adamw',
            'lr': 3e-4,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'name': 'cosine',
            'warmup_epochs': 0
        },
        'loss': {
            'name': 'bce_with_logits',
            'pos_weight_auto': True
        },
        'early_stopping': {
            'patience': 5,
            'min_delta': 0.001,
            'metric': 'val_auc',
            'mode': 'max'
        },
        'grad_clip': 1.0,
        'save_top_k': 1,
        'save_last': True
    },
    'augmentation': {
        'train': {
            'horizontal_flip': 0.5,
            'rotate_limit': 7,
            'rotate_prob': 0.3,
            'brightness_contrast': 0.3,  # Reduziert
            'hue_saturation': 0.0,  # Aus
            'blur': 0.0  # Aus
        },
        'val': {
            'enabled': False
        }
    },
    'logging': {
        'log_dir': './outputs/logs',
        'checkpoint_dir': './outputs/checkpoints',
        'visualization_dir': './outputs/visualizations',
        'log_every_n_steps': 10,
        'save_plots': True,
        'metrics': ['loss', 'auc', 'f1'],
        'domain_metrics': True
    },
    'hardware': {
        'num_threads': 4,
        'deterministic': True
    }
}

print("📋 Test Config:")
print(f"  Device: {config['experiment']['device']}")
print(f"  Batch Size: {config['training']['batch_size']}")
print(f"  Epochs: {config['training']['epochs']}")
print(f"  Input Size: {config['data']['input_size']}")
print()

try:
    print("⏳ Initialisiere Trainer...")
    trainer = Trainer(config, 'test_config')

    print("✅ Trainer erstellt!")
    print()

    print("=" * 70)
    print("🚀 STARTE TEST TRAINING (1 Epoche, 8 batch size)")
    print("=" * 70)
    print()
    print("⚠️  Drücke Ctrl+C zum Abbrechen wenn es funktioniert")
    print()

    # Starte Training
    trainer.train()

    print()
    print("=" * 70)
    print("✅ TEST ERFOLGREICH!")
    print("=" * 70)
    print()
    print("Training funktioniert! Jetzt vollständiges Training starten:")
    print("  ./scripts/train_mac.sh")
    print()

except KeyboardInterrupt:
    print()
    print("=" * 70)
    print("⚠️  Test abgebrochen (das ist OK!)")
    print("=" * 70)
    print()
    print("Wenn Training gestartet hat, funktioniert alles!")
    print("Jetzt vollständiges Training starten:")
    print("  ./scripts/train_mac.sh")
    print()

except Exception as e:
    print()
    print("=" * 70)
    print("❌ FEHLER")
    print("=" * 70)
    print()
    print(f"Fehler: {e}")
    print()
    print("Mögliche Lösungen:")
    print("1. Python scripts/test_setup.py")
    print("2. pip install -r requirements.txt")
    print("3. Siehe FIX_BUS_ERROR.md")
    print()
    sys.exit(1)
