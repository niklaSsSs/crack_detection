# 🏗️ SDNET2018 Crack Encoder Pre-training

Pre-training eines EfficientNet-B0 Encoders auf 56K Rissbildern für Transfer Learning auf Schweißnaht-Segmentierung.

**Ziel:** ROC-AUC ≥ 0.95 auf Test Set

---

## 🚀 Schnellstart

### Mac Training

```bash
# Dependencies installieren
pip install -r requirements.txt

# Training starten (5-10 Stunden)
./scripts/train_mac.sh
```

### Server Training (NVIDIA A40)

```bash
# Dependencies installieren
pip install -r requirements.txt

# Training starten (2-4 Stunden)
./scripts/train_server.sh
```

---

## 📊 Dataset

**SDNET2018**
- 56,092 Bilder (256×256 RGB)
- 3 Domänen: Bridge, Pavement, Wall
- 2 Klassen: Crack (15.1%), NonCrack (84.9%)
- 0 korrupte Dateien

Details: `data_audit.md`

---

## ⚙️ Konfigurationen

### Mac Config (`configs/config_mac.yaml`)

| Parameter | Wert |
|-----------|------|
| Device | CPU |
| Input Size | 256px |
| Batch Size | 64 |
| Epochs | 20 |
| Est. Time | 5-10h |

### Server Config (`configs/config_server_a40.yaml`)

| Parameter | Wert |
|-----------|------|
| Device | CUDA |
| Input Size | 512px |
| Batch Size | 32 |
| Epochs | 30 |
| GPU VRAM | ~18GB (40% von 46GB) |
| GPU Util | 40-60% |
| Est. Time | 2-4h |

**Server GPU Auslastung:**
- Ziel: ~40% VRAM (18GB von 46GB)
- Batch Size 32 = ~18GB VRAM
- Batch Size 24 = ~13GB VRAM (30%)
- Batch Size 48 = ~27GB VRAM (60%)

---

## 📁 Projekt-Struktur

```
crack_model/
├── configs/                # Training Configs
│   ├── config_mac.yaml
│   └── config_server_a40.yaml
│
├── src/                    # Source Code
│   ├── train.py           # Main Training Script
│   ├── dataset.py
│   └── utils.py
│
├── scripts/                # Shell Scripts
│   ├── train_mac.sh
│   └── train_server.sh
│
├── outputs/                # Training Outputs
│   ├── logs/
│   ├── checkpoints/       # ⭐ encoder_weights.pt hier!
│   └── visualizations/
│
└── SDNET2018/             # Dataset
```

---

## 📦 Output

Nach Training:

```
outputs/
└── checkpoints/crack_encoder_TIMESTAMP/
    ├── best_model.pt          # Komplettes Model
    └── encoder_weights.pt     # Nur Encoder ⭐
```

### Encoder Weights verwenden

```python
import torch
from torchvision.models import efficientnet_b0

# Laden
checkpoint = torch.load('encoder_weights.pt')
encoder = efficientnet_b0(weights=None)
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Für Segmentierung verwenden
model = UNet(encoder=encoder)

# Fine-tuning Strategy:
# 1. Freeze encoder (3-5 epochs, lr=1e-3)
# 2. Unfreeze & fine-tune (10+ epochs, lr=1e-4)
```

---

## 📈 Erwartete Performance

### Mac (20 Epochen, 256px)

- Val AUC: 0.94-0.96
- Test AUC: 0.93-0.95
- Zeit: 5-10h

### Server (30 Epochen, 512px)

- Val AUC: 0.96-0.98
- Test AUC: 0.95-0.97
- Zeit: 2-4h
- GPU: 40% VRAM

---

## 🔧 Config Anpassen

### GPU Auslastung ändern (Server)

**Weniger VRAM (20%):**
```yaml
training:
  batch_size: 16
```

**Mehr VRAM (60%):**
```yaml
training:
  batch_size: 48
```

### Training beschleunigen

**Mac:**
```yaml
training:
  epochs: 15

augmentation:
  train:
    blur: 0.0
```

**Server:**
```yaml
training:
  batch_size: 48

data:
  num_workers: 12
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```yaml
training:
  batch_size: 16  # Reduzieren
```

### Training zu langsam (Mac)

```yaml
experiment:
  device: "cpu"  # Statt mps

training:
  batch_size: 64
```

### GPU Auslastung > 50%

```yaml
training:
  batch_size: 24  # Statt 32
```

### Val AUC < 0.95

```yaml
training:
  epochs: 30
  optimizer:
    lr: 2.0e-4

data:
  input_size: 640  # Server only
```

---

## 📚 Dokumentation

- **`TRAINING_GUIDE.md`** - Detaillierte Training Anleitung
- **`PROJECT_STRUCTURE.md`** - Projekt Übersicht
- **`data_audit.md`** - Dataset Analyse
- **`configs/*.yaml`** - Config Parameter

---

## 🎯 Features

✅ **Unified Training Pipeline** - Ein Script für Mac & Server
✅ **YAML Configs** - Einfache Anpassung ohne Code
✅ **Grouped Splitting** - Verhindert Data Leakage
✅ **Class Balancing** - Weighted Sampling + Pos Weight
✅ **Domain Metrics** - Per-Domain AUC/F1 Tracking
✅ **Mixed Precision** - FP16 Training auf Server
✅ **Early Stopping** - Automatischer Stop bei Plateau
✅ **Encoder Export** - Direkt für Transfer Learning

---

## 🔬 Technische Details

**Model:**
- Backbone: EfficientNet-B0 (ImageNet)
- Parameters: ~4M
- Loss: BCEWithLogitsLoss (pos_weight=5.67)

**Optimizer:**
- AdamW (lr=3e-4, wd=1e-4)
- Cosine Annealing + Warmup
- Gradient Clipping: 1.0

**Augmentation:**
- Geometric: Flip, Rotation, Perspective
- Photometric: Brightness, Contrast, Noise, Blur
- Advanced (Server): Grid/Optical Distortion

**Data:**
- Grouped K-Fold Split (prevents leakage)
- WeightedRandomSampler (1:1 balance)
- ImageNet Normalization

---

## 📊 Monitoring

### Live Monitoring

```bash
# GPU Stats (Server)
watch -n 1 nvidia-smi

# Logs anschauen
tail -f outputs/logs/crack_encoder_*/training.log
```

### Metriken prüfen

```bash
cat outputs/logs/crack_encoder_*/test_metrics.json
```

---

## 🚦 Success Criteria

Training erfolgreich wenn:

- ✅ Val AUC ≥ 0.95
- ✅ Test AUC ≥ 0.95
- ✅ Bridge AUC ≥ 0.90
- ✅ Pavement AUC ≥ 0.90
- ✅ Wall AUC ≥ 0.90

---

## 💡 Tipps

### Mac
- Nachts laufen lassen
- `caffeinate -i ./scripts/train_mac.sh`
- Power Nap deaktivieren

### Server
- `tmux` oder `screen` verwenden
- GPU Temperature monitoren
- Andere User informieren (40% Auslastung)

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

**Minimal:**
- torch>=2.0.0
- torchvision>=0.15.0
- albumentations>=1.3.0
- scikit-learn>=1.3.0
- pyyaml

**Optional:**
- tensorboard
- wandb

---

## 🤝 Transfer Learning Workflow

```
1. Pre-training (SDNET2018 - Crack Detection)
   ↓
2. Encoder Export (encoder_weights.pt) ⭐
   ↓
3. Segmentation Model Init (U-Net/DeepLabv3+)
   ↓
4. Fine-tuning (Weld Seam Dataset)
   - Phase 1: Freeze encoder, train decoder (3-5 epochs, lr=1e-3)
   - Phase 2: Unfreeze, fine-tune all (10+ epochs, lr=1e-4)
```

---

## 📝 Citation

**Dataset:**
Dorafshan, S., Thomas, R. J., & Maguire, M. (2018). SDNET2018: An annotated image dataset for training deep learning algorithms. Data in Brief, 21, 102623.

**Model:**
Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.

---

## 📞 Support

Issues? Check:
1. **TRAINING_GUIDE.md** - Troubleshooting
2. **PROJECT_STRUCTURE.md** - Datei-Übersicht
3. **configs/*.yaml** - Parameter Docs

---

**Version:** 2.0 (Neue Struktur)
**Erstellt:** 2025-10-27
**Status:** Production Ready 🚀
