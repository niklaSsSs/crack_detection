# 📁 Projekt-Struktur

## Übersicht

```
crack_model/
│
├── SDNET2018/                      # Original Dataset (nicht im Git!)
│   ├── D/ (Bridge)
│   ├── P/ (Pavement)
│   └── W/ (Wall)
│
├── configs/                        # Konfigurations-Dateien
│   ├── config_mac.yaml            # Mac Training (CPU, 256px, 20 epochs)
│   └── config_server_a40.yaml     # Server Training (A40, 512px, 30 epochs)
│
├── src/                            # Source Code
│   ├── train.py                   # Haupt-Training Script
│   ├── dataset.py                 # Dataset & Data Loading
│   └── utils.py                   # Hilfsfunktionen
│
├── scripts/                        # Shell Scripts
│   ├── train_mac.sh               # Training auf Mac starten
│   └── train_server.sh            # Training auf Server starten
│
├── outputs/                        # Training Outputs
│   ├── logs/                      # Training Logs
│   │   └── crack_encoder_*/
│   │       ├── config.yaml        # Verwendete Config
│   │       └── test_metrics.json  # Test Metriken
│   │
│   ├── checkpoints/               # Model Checkpoints
│   │   └── crack_encoder_*/
│   │       ├── best_model.pt      # Bestes Model
│   │       └── encoder_weights.pt # Encoder für Transfer ⭐
│   │
│   └── visualizations/            # Plots & Visualisierungen
│       └── crack_encoder_*/
│           ├── confusion_matrix.png
│           └── training_history.png
│
├── data/                           # (Optional) Preprocessed Data
├── models/                         # (Optional) Alternative Models
├── notebooks/                      # Jupyter Notebooks
│
├── data_audit.py                   # Dataset Audit Script (alt)
├── data_audit.md                   # Audit Report
├── data_audit.json                 # Audit Report (JSON)
├── samples_*.png                   # Sample Thumbnails (6 files)
│
├── README.md                       # Hauptdokumentation
├── PROJECT_STRUCTURE.md            # Diese Datei
├── TRAINING_GUIDE.md               # Training Anleitung
└── requirements.txt                # Python Dependencies
```

---

## 📄 Wichtige Dateien

### Configs

| Datei | Beschreibung | Hardware | Input Size | Batch Size |
|-------|-------------|----------|------------|------------|
| `config_mac.yaml` | Mac Training | CPU/MPS | 256px | 64 |
| `config_server_a40.yaml` | Server Training | A40 GPU | 512px | 32 |

### Training Scripts

| Datei | Verwendung |
|-------|-----------|
| `src/train.py` | Haupt-Training Script (beide Configs) |
| `scripts/train_mac.sh` | Mac Training starten |
| `scripts/train_server.sh` | Server Training starten |

### Outputs

| Verzeichnis | Inhalt |
|-------------|--------|
| `outputs/logs/` | Training Logs, Config, Metriken |
| `outputs/checkpoints/` | Model Checkpoints, **Encoder Weights** ⭐ |
| `outputs/visualizations/` | Plots, Confusion Matrix |

---

## 🚀 Schnellstart

### Mac Training

```bash
./scripts/train_mac.sh
```

**Oder manuell:**

```bash
python src/train.py --config configs/config_mac.yaml
```

### Server Training

```bash
./scripts/train_server.sh
```

**Oder manuell:**

```bash
python src/train.py --config configs/config_server_a40.yaml
```

---

## 📊 Config Parameter

### Mac Config (`config_mac.yaml`)

**Optimiert für:**
- Apple Silicon Macs (CPU Modus)
- Schnelles lokales Training
- Niedrige VRAM Anforderungen

**Key Settings:**
- Device: `cpu`
- Input Size: `256px` (native)
- Batch Size: `64`
- Epochs: `20`
- Mixed Precision: `false`
- Augmentations: Moderat

**Geschätzte Zeit:** 5-10 Stunden

### Server Config (`config_server_a40.yaml`)

**Optimiert für:**
- NVIDIA A40 GPU (46GB VRAM)
- ~40% GPU Auslastung (~18GB VRAM)
- Hohe Qualität & Accuracy

**Key Settings:**
- Device: `cuda`
- Input Size: `512px` (upscaled)
- Batch Size: `32`
- Epochs: `30`
- Mixed Precision: `true` (FP16)
- Augmentations: Aggressiv
- Gradient Accumulation: `2` (effektiv 64 samples)

**GPU Auslastung:**
- VRAM: ~18-20GB von 46GB (≈40%)
- GPU Util: ~40-60%
- Power: ~150-180W von 300W

**Geschätzte Zeit:** 2-4 Stunden

---

## 🔧 Config Anpassen

### Batch Size ändern (GPU Auslastung)

**Weniger VRAM verwenden (z.B. 20%):**

```yaml
training:
  batch_size: 16  # Statt 32
```

**Mehr VRAM verwenden (z.B. 60%):**

```yaml
training:
  batch_size: 48  # Statt 32
```

**Formel:**
- EfficientNet-B0 @ 512px: ~0.6GB pro 32 samples
- VRAM Usage ≈ `batch_size × 0.6GB / 32`

### Augmentations anpassen

**Mehr Augmentation:**

```yaml
augmentation:
  train:
    horizontal_flip: 0.7      # Statt 0.5
    brightness_contrast: 0.7  # Statt 0.5
    # ...
```

**Weniger Augmentation (schneller):**

```yaml
augmentation:
  train:
    horizontal_flip: 0.5
    brightness_contrast: 0.3
    blur: 0.0  # Deaktivieren
```

---

## 📈 Outputs verstehen

### Logs Verzeichnis

```
outputs/logs/crack_encoder_mac_20251027_123456/
├── config.yaml           # Verwendete Konfiguration
└── test_metrics.json     # Test Ergebnisse
```

**`test_metrics.json` Beispiel:**

```json
{
  "loss": 0.0234,
  "auc": 0.9678,
  "f1": 0.8934,
  "precision": 0.9123,
  "recall": 0.8756,
  "confusion_matrix": [[2200, 272], [98, 382]],
  "domain_metrics": {
    "Bridge": {"auc": 0.9543, "f1": 0.8712, "n_samples": 504},
    "Pavement": {"auc": 0.9701, "f1": 0.8934, "n_samples": 936},
    "Wall": {"auc": 0.9723, "f1": 0.9123, "n_samples": 1512}
  }
}
```

### Checkpoints Verzeichnis

```
outputs/checkpoints/crack_encoder_mac_20251027_123456/
├── best_model.pt         # Komplettes Model (>15MB)
└── encoder_weights.pt    # Nur Encoder (>15MB) ⭐
```

**`encoder_weights.pt` verwenden:**

```python
import torch
from torchvision.models import efficientnet_b0

# Laden
checkpoint = torch.load('encoder_weights.pt')
encoder = efficientnet_b0(weights=None)
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Config Info
config = checkpoint['config']
print(f"Input Size: {config['input_size']}")
print(f"Best Val AUC: {config['best_val_auc']:.4f}")
```

### Visualizations Verzeichnis

```
outputs/visualizations/crack_encoder_mac_20251027_123456/
├── confusion_matrix.png    # 2x2 Matrix (Crack/NonCrack)
└── training_history.png    # 4 Plots (Loss, AUC, F1, LR)
```

---

## 🛠️ Entwicklung

### Neues Augmentation hinzufügen

1. **Config erweitern** (`config_*.yaml`):

```yaml
augmentation:
  train:
    cutout: 0.3  # Neu
```

2. **Code erweitern** (`src/train.py`):

```python
if aug_cfg.get('cutout', 0) > 0:
    transforms.append(A.CoarseDropout(p=aug_cfg['cutout']))
```

### Neues Backbone verwenden

1. **Config ändern** (`config_*.yaml`):

```yaml
model:
  backbone: "efficientnet_b1"  # Statt b0
```

2. **Code anpassen** (`src/train.py`):

```python
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

# In CrackClassifier.__init__:
if config['model']['pretrained']:
    weights = EfficientNet_B1_Weights.IMAGENET1K_V1
    self.encoder = efficientnet_b1(weights=weights)
else:
    self.encoder = efficientnet_b1(weights=None)
```

---

## 📝 Best Practices

### Mac Training

✅ **Do:**
- `device: cpu` verwenden (schneller als MPS)
- `batch_size: 64` für gute CPU Auslastung
- `input_size: 256` (native, kein Upscaling)
- `num_workers: 4` (an CPU Cores anpassen)

❌ **Don't:**
- `device: mps` verwenden (zu langsam für EfficientNet)
- `batch_size < 32` (ineffizient)
- `input_size: 512` (4x langsamer, wenig Benefit)

### Server Training

✅ **Do:**
- `mixed_precision: true` (2x schneller)
- `batch_size: 32` für stabile 40% Auslastung
- `gradient_accumulation_steps: 2` (effektiv 64)
- `persistent_workers: true` (schnelleres Data Loading)

❌ **Don't:**
- `batch_size > 64` (VRAM Limit, OOM Risk)
- `num_workers > 12` (diminishing returns)
- `deterministic: true` (langsamer, nur für Debugging)

---

## 🐛 Troubleshooting

### Problem: CUDA Out of Memory

**Lösung:**

```yaml
training:
  batch_size: 16  # Reduzieren
```

### Problem: Training zu langsam auf Mac

**Lösung:**

```yaml
experiment:
  device: "cpu"  # Statt mps

training:
  batch_size: 64  # Erhöhen
```

### Problem: GPU Auslastung zu niedrig

**Lösung:**

```yaml
training:
  batch_size: 48  # Erhöhen

data:
  num_workers: 8  # Erhöhen
```

### Problem: GPU Auslastung zu hoch (>50%)

**Lösung:**

```yaml
training:
  batch_size: 24  # Reduzieren

performance:
  gradient_accumulation_steps: 3  # Erhöhen
```

---

## 📚 Weitere Dokumentation

- **`README.md`** - Hauptdokumentation
- **`TRAINING_GUIDE.md`** - Detaillierte Training Anleitung
- **`data_audit.md`** - Dataset Analyse
- **`configs/*.yaml`** - Inline Kommentare

---

**Erstellt:** 2025-10-27
**Version:** 2.0 (Neue Struktur)
