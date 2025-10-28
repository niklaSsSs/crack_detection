#!/bin/bash

# Training Script für NVIDIA A40 Server

cd "$(dirname "$0")/.." || exit

echo "======================================"
echo "Training auf NVIDIA A40 Server"
echo "======================================"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi nicht gefunden"
else
    echo ""
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python nicht gefunden"
    exit 1
fi

# Check CUDA in PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA nicht verfügbar!'"
if [ $? -ne 0 ]; then
    echo "Error: CUDA in PyTorch nicht verfügbar"
    exit 1
fi

echo "PyTorch CUDA Version: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Run training
python src/train.py --config configs/config_server_a40.yaml

echo ""
echo "======================================"
echo "Training abgeschlossen!"
echo "======================================"
echo ""
echo "GPU Auslastung während Training:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader
echo ""
echo "Ergebnisse in: outputs/"
echo "Encoder Weights: outputs/checkpoints/.../encoder_weights.pt"
