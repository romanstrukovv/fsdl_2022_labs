#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

# 1. Disable W&B completely to run offline without credentials or login prompts
export WANDB_MODE=disabled
export WANDB_SILENT=true

# 2. Define Google Drive source and lab artifact destination
DRIVE_MODEL_PATH="${DRIVE_MODEL_PATH:-/content/drive/MyDrive/model.pt}"
STAGED_MODEL_NAME="${STAGED_MODEL_NAME:-paragraph-text-recognizer}"
TARGET_DIR="text_recognizer/artifacts/${STAGED_MODEL_NAME}"
TARGET_FILE="${TARGET_DIR}/model.pt"

echo "=== 1. Checking Google Drive source ==="
if [ ! -f "$DRIVE_MODEL_PATH" ]; then
  echo "Error: Model file not found at $DRIVE_MODEL_PATH"
  echo "Ensure Google Drive is mounted and 'model.pt' exists in your MyDrive root."
  exit 1
fi
echo "Found model in Google Drive: $DRIVE_MODEL_PATH"

echo "=== 2. Training small sanity model locally (no W&B) ==="
# Removed --wandb flag so PyTorch Lightning logs locally without W&B API calls
python training/run_experiment.py --data_class=IAMParagraphs --model_class=ResnetTransformer --loss=transformer \
  --tf_dim 4 --tf_fc_dim 2 --tf_layers 2 --tf_nhead 2 --batch_size 2 --lr 0.0001 \
  --limit_train_batches 1 --limit_val_batches 1 --limit_test_batches 1 --num_sanity_val_steps 0 \
  --num_workers 1 || FAILURE=true

echo "=== 3. Fetching model from mounted Google Drive ==="
mkdir -p "$TARGET_DIR"
cp "$DRIVE_MODEL_PATH" "$TARGET_FILE" || FAILURE=true

if [ -f "$TARGET_FILE" ]; then
  FILE_SIZE=$(du -h "$TARGET_FILE" | cut -f1)
  echo "Model successfully copied to $TARGET_FILE (size: $FILE_SIZE)"
else
  echo "Error: Failed to copy model file to $TARGET_FILE"
  FAILURE=true
fi

echo "=== 4. Verifying TorchScript model integrity ==="
python -c "
import torch
try:
    model = torch.jit.load('$TARGET_FILE')
    model.eval()
    print('Verification passed: TorchScript model loaded cleanly.')
except Exception as e:
    print(f'Verification failed: {e}')
    exit(1)
" || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Model staging test failed."
  exit 1
fi

echo "Model staging test passed successfully!"
exit 0