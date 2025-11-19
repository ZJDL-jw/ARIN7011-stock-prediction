#!/bin/bash
# Evaluate all models with optimized thresholds

echo "=== Evaluating All Models with Optimal Thresholds ==="
echo ""

# Evaluate each model
echo "1. Evaluating MLP..."
python src/evaluate.py --config configs/mlp.yaml

echo ""
echo "2. Evaluating LSTM..."
python src/evaluate.py --config configs/lstm.yaml

echo ""
echo "3. Evaluating GRU..."
python src/evaluate.py --config configs/gru.yaml

echo ""
echo "4. Evaluating CNN1D..."
python src/evaluate.py --config configs/cnn1d.yaml

echo ""
echo "5. Evaluating PatchTST..."
python src/evaluate.py --config configs/patchtst.yaml

echo ""
echo "6. Evaluating SimpleTransformer..."
python src/evaluate.py --config configs/simple_transformer.yaml

echo ""
echo "7. Evaluating Transformer..."
python src/evaluate.py --config configs/transformer.yaml

echo ""
echo "8. Evaluating Ensemble (with optimized weights)..."
python src/ensemble.py --config configs/lstm.yaml --models lstm gru patchtst --method weighted

echo ""
echo "=== Evaluation Complete ==="

