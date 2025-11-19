#!/bin/bash
# Train all models with optimized settings

echo "=== Training All Models with Optimizations ==="
echo ""

# Train each model
echo "1. Training MLP..."
python src/train.py --config configs/mlp.yaml

echo ""
echo "2. Training LSTM..."
python src/train.py --config configs/lstm.yaml

echo ""
echo "3. Training GRU..."
python src/train.py --config configs/gru.yaml

echo ""
echo "4. Training CNN1D..."
python src/train.py --config configs/cnn1d.yaml

echo ""
echo "5. Training PatchTST..."
python src/train.py --config configs/patchtst.yaml

echo ""
echo "6. Training SimpleTransformer..."
python src/train.py --config configs/simple_transformer.yaml

echo ""
echo "7. Training Transformer..."
python src/train.py --config configs/transformer.yaml

echo ""
echo "=== Training Complete ==="

