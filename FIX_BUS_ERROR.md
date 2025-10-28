# 🔧 Bus Error Fix für Mac Training

## ❌ Problem: Bus Error beim Training

**Symptome:**
- Training startet, läuft 1-2 Batches
- Dann: "Bus error" oder einfach Absturz
- Sehr langsam: ~70-80 Sekunden pro Batch

**Ursache:**
- **MPS (Apple Silicon GPU)** ist extrem langsam für EfficientNet
- PyTorch MPS hat Bugs bei komplexen Models
- Führt zu Memory-Korruption → Bus Error

---

## ✅ Lösung: CPU verwenden

### Option 1: Neue Config verwenden (empfohlen)

Die `config_mac.yaml` ist bereits auf CPU eingestellt:

```bash
# Training mit neuer Config
python src/train.py --config configs/config_mac.yaml
```

Oder mit Shell Script:
```bash
./scripts/train_mac.sh
```

**Vorteile CPU:**
- ✅ Stabil, keine Bus Errors
- ✅ Schneller als MPS (~5-10 Sekunden pro Batch statt 70)
- ✅ Nutzt alle CPU Cores

---

### Option 2: Alte Scripts anpassen

Falls du die alten Scripts verwendest:

**`train_encoder_cpu.py` bereits auf CPU**
```bash
python archive/train_encoder_cpu.py
```

**`train_encoder.py` umstellen:**
```python
# Zeile ~42 ändern
DEVICE = torch.device('cpu')  # Statt 'mps'
```

---

## 📊 Erwartete Performance (CPU)

### Mit neuer Config (`config_mac.yaml`)

- **Batch Size:** 64
- **Input Size:** 256px
- **Zeit pro Batch:** ~5-10 Sekunden
- **Zeit pro Epoch:** ~2-3 Stunden
- **Total (20 Epochen):** ~40-60 Stunden

### Tipps für schnelleres Training

**1. Reduziere Epochen:**
```yaml
# configs/config_mac.yaml
training:
  epochs: 10  # Statt 20
```

**2. Deaktiviere langsame Augmentationen:**
```yaml
augmentation:
  train:
    blur: 0.0  # Deaktivieren
    hue_saturation: 0.0
```

**3. Erhöhe Batch Size (falls genug RAM):**
```yaml
training:
  batch_size: 128  # Statt 64
```

---

## 🐛 Andere Bus Error Ursachen

Falls Bus Error weiterhin auftritt:

### 1. RAM Problem

**Check RAM:**
```bash
# Activity Monitor öffnen
open -a "Activity Monitor"

# Oder im Terminal
vm_stat
```

**Lösung:** Batch Size reduzieren
```yaml
training:
  batch_size: 32  # Statt 64
```

### 2. Korrupte Bilder im Dataset

**Check:**
```bash
python scripts/data_audit.py
# Siehe data_audit.md für corrupt files
```

### 3. PyTorch Installation Problem

**Reinstall:**
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

---

## 🚀 Empfohlener Workflow (Mac)

```bash
# 1. Alte Prozesse killen
pkill -f train_encoder

# 2. Setup prüfen
python scripts/test_setup.py

# 3. Training mit CPU Config
./scripts/train_mac.sh

# Oder direkt:
python src/train.py --config configs/config_mac.yaml

# 4. Monitoring
# In separatem Terminal:
tail -f outputs/logs/crack_encoder_mac_*/training.log
```

---

## ⚡ Schnelles Test-Training

Für Test (1 Epoche, nur 1000 Bilder):

```python
# Erstelle test_config.yaml
cp configs/config_mac.yaml configs/test_config.yaml

# Editiere:
training:
  epochs: 1

# Dann:
python src/train.py --config configs/test_config.yaml
```

---

## 📊 Vergleich: MPS vs CPU

| | MPS (Apple GPU) | CPU |
|---|---|---|
| **Geschwindigkeit** | 70-80s/batch | 5-10s/batch |
| **Stabilität** | ❌ Bus Errors | ✅ Stabil |
| **RAM Usage** | Hoch | Mittel |
| **Empfohlen?** | ❌ NEIN | ✅ JA |

**Fazit:** CPU ist für EfficientNet auf Mac M-Chips **schneller UND stabiler**!

---

## 🆘 Immer noch Probleme?

### Quick Fix:

```bash
# 1. Alles stoppen
pkill -f python

# 2. Neustarten
cd /Users/niklaswegner/python_proj/crack_model

# 3. Mit reduzierter Config
python src/train.py --config configs/config_mac.yaml
```

### Logs checken:

```bash
# Letzte Fehler
tail -100 outputs/logs/crack_encoder_mac_*/training.log

# System Logs
log show --predicate 'eventMessage contains "bus error"' --last 1h
```

---

**Die Config ist bereits auf CPU eingestellt - einfach `./scripts/train_mac.sh` starten!**
