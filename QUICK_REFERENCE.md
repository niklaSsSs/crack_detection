# ⚡ Quick Reference

## 🚀 Training Starten

### Mac
```bash
./scripts/train_mac.sh
```

### Server (A40)
```bash
./scripts/train_server.sh
```

---

## 🔍 Setup & Monitoring

### Setup prüfen
```bash
python scripts/test_setup.py
```

### GPU Stats (Server)
```bash
# Live Monitoring
watch -n 1 nvidia-smi

# Nur wichtige Infos
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

### Training Logs
```bash
# Live Logs
tail -f outputs/logs/crack_encoder_*/training.log

# Test Metriken
cat outputs/logs/crack_encoder_*/test_metrics.json | python -m json.tool
```

---

## ⚙️ Config Anpassen

### GPU Auslastung ändern (Server)

**20% VRAM:**
```yaml
# configs/config_server_a40.yaml
training:
  batch_size: 16
```

**40% VRAM (Standard):**
```yaml
training:
  batch_size: 32
```

**60% VRAM:**
```yaml
training:
  batch_size: 48
```

### Training beschleunigen (Mac)

```yaml
# configs/config_mac.yaml
training:
  epochs: 15  # Statt 20

augmentation:
  train:
    blur: 0.0  # Deaktivieren
```

### Höhere Accuracy (Server)

```yaml
# configs/config_server_a40.yaml
data:
  input_size: 640  # Statt 512

training:
  epochs: 40  # Statt 30
```

---

## 📦 Encoder Weights verwenden

```python
import torch
from torchvision.models import efficientnet_b0

# Laden
checkpoint = torch.load('outputs/checkpoints/.../encoder_weights.pt')
encoder = efficientnet_b0(weights=None)
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Config Info
config = checkpoint['config']
print(f"Input Size: {config['input_size']}")
print(f"Best Val AUC: {config['best_val_auc']:.4f}")
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
```

### GPU Auslastung > 50%
```yaml
training:
  batch_size: 24  # Statt 32
```

---

## 📊 Batch Size Guide (Server A40)

| Batch | VRAM | % |
|-------|------|---|
| 16 | ~9GB | 20% |
| 24 | ~13GB | 30% |
| **32** | **~18GB** | **40%** ← Standard |
| 48 | ~27GB | 60% |

---

## 📁 Wichtige Dateien

| Datei | Beschreibung |
|-------|-------------|
| `configs/config_mac.yaml` | Mac Training Config |
| `configs/config_server_a40.yaml` | Server Training Config |
| `outputs/checkpoints/.../encoder_weights.pt` | Encoder für Transfer ⭐ |
| `outputs/logs/.../test_metrics.json` | Test Ergebnisse |

---

## 📚 Dokumentation

| Datei | Inhalt |
|-------|--------|
| `README_NEW.md` | Hauptdokumentation |
| `TRAINING_GUIDE.md` | Detaillierte Anleitung |
| `PROJECT_STRUCTURE.md` | Struktur Details |
| `NEUE_STRUKTUR_SUMMARY.md` | Zusammenfassung Änderungen |

---

## ✅ Success Criteria

- Val AUC ≥ 0.95
- Test AUC ≥ 0.95
- Bridge AUC ≥ 0.90
- Pavement AUC ≥ 0.90
- Wall AUC ≥ 0.90

---

## 🎯 Typischer Workflow

### Mac

```bash
# 1. Setup prüfen
python scripts/test_setup.py

# 2. Training starten (5-10h)
./scripts/train_mac.sh

# 3. Ergebnisse prüfen
cat outputs/logs/crack_encoder_mac_*/test_metrics.json
```

### Server

```bash
# 1. Setup prüfen
python scripts/test_setup.py

# 2. GPU prüfen
nvidia-smi

# 3. Training starten (2-4h)
./scripts/train_server.sh

# 4. Monitoring (separates Terminal)
watch -n 1 nvidia-smi

# 5. Ergebnisse prüfen
cat outputs/logs/crack_encoder_a40_*/test_metrics.json
```

---

**Version:** 2.0
**Erstellt:** 2025-10-27
