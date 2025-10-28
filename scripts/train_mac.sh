#!/bin/bash

# Training Script für Mac (CPU/MPS)

cd "$(dirname "$0")/.." || exit

echo "======================================"
echo "Training auf Mac (CPU optimiert)"
echo "======================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python nicht gefunden"
    exit 1
fi

# Check dependencies
python -c "import torch; import albumentations; import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Dependencies fehlen. Bitte installieren:"
    echo "  pip install torch torchvision albumentations scikit-learn pyyaml matplotlib seaborn"
    exit 1
fi

# Run training
python src/train.py --config configs/config_mac.yaml

echo ""
echo "======================================"
echo "Training abgeschlossen!"
echo "======================================"
echo ""
echo "Ergebnisse in: outputs/"
echo "Encoder Weights: outputs/checkpoints/.../encoder_weights.pt"
