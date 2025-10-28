# 🎯 START HERE

## ✅ Projekt ist bereit!

Die Struktur wurde aufgeräumt und ist production-ready.

---

## 🚀 Training Starten

### 1. Setup prüfen

```bash
python scripts/test_setup.py
```

### 2. Training starten

**Mac (CPU, 5-10 Stunden):**
```bash
./scripts/train_mac.sh
```

**Server (A40 GPU, 2-4 Stunden, 40% VRAM):**
```bash
./scripts/train_server.sh
```

---

## 📖 Dokumentation

| Datei | Inhalt |
|-------|--------|
| **README.md** | ⭐ Start hier! Hauptdokumentation |
| **QUICK_REFERENCE.md** | Schnellreferenz für häufige Befehle |
| **TRAINING_GUIDE.md** | Detaillierte Training Anleitung |
| **PROJECT_STRUCTURE.md** | Struktur Details |

---

## ⚙️ Konfiguration

| Config | Hardware | Input | Batch | Zeit |
|--------|----------|-------|-------|------|
| `configs/config_mac.yaml` | CPU | 256px | 64 | 5-10h |
| `configs/config_server_a40.yaml` | A40 | 512px | 32 | 2-4h |

**Server GPU Auslastung:** ~40% VRAM (18GB von 46GB)

Anpassen:
- `batch_size: 16` → 20% VRAM (~9GB)
- `batch_size: 32` → 40% VRAM (~18GB) ← Standard
- `batch_size: 48` → 60% VRAM (~27GB)

---

## 📁 Wichtige Ordner

```
crack_model/
├── configs/        Training Configs (YAML)
├── src/            Source Code (Python)
├── scripts/        Ausführbare Scripts
├── outputs/        Training Results (wird erstellt)
│   └── checkpoints/encoder_weights.pt ⭐
├── docs/           Dokumentation & Referenzen
└── archive/        Alte Scripts (nicht mehr nötig)
```

---

## 📊 Nach dem Training

**Encoder Weights finden:**
```
outputs/checkpoints/crack_encoder_TIMESTAMP/encoder_weights.pt
```

**Verwendung:**
```python
import torch
from torchvision.models import efficientnet_b0

checkpoint = torch.load('encoder_weights.pt')
encoder = efficientnet_b0(weights=None)
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Für Segmentierung verwenden
model = UNet(encoder=encoder)
```

---

## 🎯 Success Criteria

- ✅ Val AUC ≥ 0.95
- ✅ Test AUC ≥ 0.95
- ✅ Alle Domain AUCs ≥ 0.90

---

## 🆘 Hilfe

**Setup Probleme:**
```bash
python scripts/test_setup.py
```

**CUDA Out of Memory:**
```yaml
# configs/config_server_a40.yaml
training:
  batch_size: 16  # Statt 32
```

**Training zu langsam (Mac):**
```yaml
# configs/config_mac.yaml
experiment:
  device: "cpu"  # Statt mps
```

**GPU Auslastung > 50%:**
```yaml
# configs/config_server_a40.yaml
training:
  batch_size: 24  # Statt 32
```

---

## 📚 Nächste Schritte

1. **Setup prüfen:** `python scripts/test_setup.py`
2. **README lesen:** `cat README.md`
3. **Training starten:** `./scripts/train_mac.sh` oder `./scripts/train_server.sh`
4. **Monitoring:** `watch -n 1 nvidia-smi` (Server only)
5. **Ergebnisse prüfen:** `cat outputs/logs/.../test_metrics.json`

---

**Viel Erfolg! 🚀**
