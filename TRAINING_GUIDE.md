# 🚀 Training Guide

## Schnellstart

### Mac (CPU)

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. Training starten
./scripts/train_mac.sh

# Oder manuell:
python src/train.py --config configs/config_mac.yaml
```

**Dauer:** ~5-10 Stunden
**Output:** `outputs/checkpoints/.../encoder_weights.pt`

---

### Server (NVIDIA A40)

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. CUDA Check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Training starten
./scripts/train_server.sh

# Oder manuell:
python src/train.py --config configs/config_server_a40.yaml
```

**Dauer:** ~2-4 Stunden
**GPU Auslastung:** ~40% VRAM (18GB von 46GB)
**Output:** `outputs/checkpoints/.../encoder_weights.pt`

---

## 📊 Training überwachen

### Live Monitoring (während Training)

**Terminal 1 (Training):**
```bash
python src/train.py --config configs/config_mac.yaml
```

**Terminal 2 (Monitoring):**
```bash
# GPU Stats (Server)
watch -n 1 nvidia-smi

# CPU Stats (Mac)
top -pid $(pgrep -f train.py)
```

### Logs checken

```bash
# Neueste Logs anzeigen
tail -f outputs/logs/crack_encoder_*/training.log

# Metriken anschauen
cat outputs/logs/crack_encoder_*/test_metrics.json | python -m json.tool
```

---

## 🔧 Config Anpassungen

### GPU Auslastung ändern

**Aktuelle Auslastung:** ~40% VRAM auf A40 (18GB von 46GB)

**Weniger VRAM (20% - 9GB):**

```yaml
# configs/config_server_a40.yaml
training:
  batch_size: 16  # Statt 32
```

**Mehr VRAM (60% - 28GB):**

```yaml
training:
  batch_size: 48  # Statt 32
```

**VRAM Berechnung:**
- Batch 16 @ 512px: ~9GB (20%)
- Batch 32 @ 512px: ~18GB (40%) ← **Standard**
- Batch 48 @ 512px: ~27GB (60%)
- Batch 64 @ 512px: ~36GB (78%)

### Training beschleunigen

**Mac (CPU):**

```yaml
# Weniger Augmentations
augmentation:
  train:
    brightness_contrast: 0.3  # Statt 0.5
    blur: 0.0                  # Deaktivieren

# Weniger Epochen
training:
  epochs: 15  # Statt 20
```

**Server (GPU):**

```yaml
# Größere Batch Size
training:
  batch_size: 48  # Statt 32

# Mehr Workers
data:
  num_workers: 12  # Statt 8

# Benchmark Mode
hardware:
  benchmark: true
  deterministic: false
```

### Accuracy verbessern

```yaml
# Höhere Auflösung
data:
  input_size: 640  # Statt 512 (Server only!)

# Mehr Epochen
training:
  epochs: 40  # Statt 30

# Mehr Augmentation
augmentation:
  train:
    horizontal_flip: 0.7
    vertical_flip: 0.3
    rotate_prob: 0.5
```

---

## 📈 Erwartete Performance

### Mac Training (20 Epochen, 256px)

| Metrik | Erwartung |
|--------|-----------|
| Val AUC | 0.94-0.96 |
| Test AUC | 0.93-0.95 |
| Test F1 | 0.85-0.90 |
| Training Zeit | 5-10h |

### Server Training (30 Epochen, 512px)

| Metrik | Erwartung |
|--------|-----------|
| Val AUC | 0.96-0.98 |
| Test AUC | 0.95-0.97 |
| Test F1 | 0.88-0.92 |
| Training Zeit | 2-4h |
| GPU VRAM | ~18GB |
| GPU Util | 40-60% |

---

## 🎯 Success Criteria

Training ist erfolgreich wenn:

✅ **Val AUC ≥ 0.95**
✅ **Test AUC ≥ 0.95**
✅ **Alle Domain AUCs ≥ 0.90**
- Bridge AUC ≥ 0.90
- Pavement AUC ≥ 0.90
- Wall AUC ≥ 0.90

---

## 📦 Outputs verwenden

### Encoder Weights laden

```python
import torch
from torchvision.models import efficientnet_b0

# Laden
checkpoint = torch.load('outputs/checkpoints/.../encoder_weights.pt')
encoder = efficientnet_b0(weights=None)
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Info
config = checkpoint['config']
print(f"Pretrained on: {config['pretrained_on']}")
print(f"Input size: {config['input_size']}px")
print(f"Best Val AUC: {config['best_val_auc']:.4f}")
print(f"Normalization: mean={config['mean']}, std={config['std']}")
```

### Für Segmentierung verwenden

```python
# U-Net Beispiel
class UNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # Decoder bauen
        self.decoder = nn.Sequential(
            # ... Decoder Layers
        )

    def forward(self, x):
        # Encoder Features
        features = self.encoder(x)

        # Decoder
        output = self.decoder(features)
        return output

# Initialisieren mit pre-trained Encoder
encoder = load_encoder('encoder_weights.pt')
model = UNet(encoder)

# Fine-tuning Strategy:
# 1. Freeze encoder (3-5 epochs)
for param in encoder.parameters():
    param.requires_grad = False

# Train decoder only
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3)

# 2. Unfreeze and fine-tune (10+ epochs)
for param in encoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

---

## 🐛 Troubleshooting

### Training startet nicht

**Problem:** Import Fehler
```
ModuleNotFoundError: No module named 'albumentations'
```

**Lösung:**
```bash
pip install -r requirements.txt
```

---

### CUDA Out of Memory (Server)

**Problem:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Lösung 1:** Batch Size reduzieren
```yaml
training:
  batch_size: 16  # Statt 32
```

**Lösung 2:** Gradient Accumulation erhöhen
```yaml
performance:
  gradient_accumulation_steps: 4  # Statt 2
```

**Lösung 3:** Input Size reduzieren
```yaml
data:
  input_size: 384  # Statt 512
```

---

### Training zu langsam (Mac)

**Problem:** MPS langsamer als erwartet

**Lösung:** CPU verwenden
```yaml
experiment:
  device: "cpu"  # Statt mps

training:
  batch_size: 64  # Größer für CPU
```

---

### Validation AUC < 0.95

**Problem:** Performance zu niedrig

**Lösung 1:** Mehr Epochen
```yaml
training:
  epochs: 30  # Statt 20
```

**Lösung 2:** Learning Rate anpassen
```yaml
training:
  optimizer:
    lr: 2.0e-4  # Statt 3.0e-4 (kleiner = stabiler)
```

**Lösung 3:** Mehr Augmentation
```yaml
augmentation:
  train:
    horizontal_flip: 0.7
    brightness_contrast: 0.6
```

**Lösung 4:** Höhere Auflösung (Server)
```yaml
data:
  input_size: 640  # Statt 512
```

---

### GPU Auslastung zu hoch (>50%)

**Problem:** Server soll nur 40% nutzen

**Lösung:**
```yaml
training:
  batch_size: 24  # Statt 32 (von 40% auf ~30%)
```

Oder:

```yaml
training:
  batch_size: 20  # Noch weniger (von 40% auf ~25%)
```

**Monitoring:**
```bash
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader'
```

---

## 📊 Monitoring Tools

### TensorBoard (Optional)

**1. Config aktivieren:**
```yaml
monitoring:
  use_tensorboard: true
```

**2. TensorBoard starten:**
```bash
tensorboard --logdir outputs/logs
```

**3. Browser öffnen:**
```
http://localhost:6006
```

### Weights & Biases (Optional)

**1. Config aktivieren:**
```yaml
monitoring:
  use_wandb: true
  project_name: "crack-detection"
```

**2. Login:**
```bash
wandb login
```

**3. Training startet automatisch Tracking**

---

## 🔄 Resume Training

Falls Training abbricht:

**1. Checkpoint prüfen:**
```bash
ls -lh outputs/checkpoints/crack_encoder_*/best_model.pt
```

**2. Training fortsetzen:**
```python
# TODO: Resume Funktion implementieren
# Aktuell: Training startet neu
```

---

## 📝 Tipps & Best Practices

### Mac Training

✅ **Empfehlungen:**
- Nachts laufen lassen (5-10h)
- Power Nap deaktivieren
- Terminal offen lassen
- `caffeinate` verwenden:
  ```bash
  caffeinate -i ./scripts/train_mac.sh
  ```

❌ **Vermeiden:**
- Parallele CPU-intensive Tasks
- Safari mit vielen Tabs
- Video Encoding während Training

### Server Training

✅ **Empfehlungen:**
- `tmux` oder `screen` verwenden
- GPU Temperature monitoren
- Logs regelmäßig checken
- Andere User informieren (40% Auslastung)

❌ **Vermeiden:**
- Batch Size > 64 (VRAM Limit)
- Mehrere Trainings parallel
- GPU ohne Monitoring

### Allgemein

✅ **Best Practices:**
- Config committen (Git)
- Outputs regelmäßig sichern
- Metriken dokumentieren
- Encoder Weights umbenennen mit AUC:
  ```bash
  mv encoder_weights.pt encoder_auc_0.9678.pt
  ```

---

## 📚 Weitere Ressourcen

- **PROJECT_STRUCTURE.md** - Projekt Übersicht
- **README.md** - Hauptdokumentation
- **data_audit.md** - Dataset Analyse
- **configs/config_*.yaml** - Config Parameter

---

**Viel Erfolg beim Training! 🚀**
