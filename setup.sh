#!/bin/bash

set -e

echo "🚀 Setting up Mistral-7B Fine-tuning Environment"
echo "================================================"

VENV_NAME="mistral_finetune"
DATA_SOURCE="toor@20.163.60.124:/home/toor/synth/formatted_data.jsonl"
DATA_FILE="training_data.jsonl"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
LOG_FILE="logs/training.log"

echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev build-essential

echo "📦 Creating virtual environment..."
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

echo "⚡ Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate bitsandbytes
pip install deepspeed wandb tqdm

echo "📁 Creating directories..."
mkdir -p data models logs

echo "📥 Downloading training data..."
echo "Copying training data from $DATA_SOURCE..."
scp $DATA_SOURCE ./data/$DATA_FILE

echo "🤖 Downloading base model..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading Mistral-7B-Instruct model...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_NAME',
    torch_dtype=torch.float16,
    device_map='auto'
)

print('Saving model locally...')
tokenizer.save_pretrained('./models/mistral-7b-instruct')
model.save_pretrained('./models/mistral-7b-instruct')
print('✅ Model downloaded and saved!')
"

echo "🔐 Setting up wandb..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "Using WANDB_API_KEY environment variable"
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "Please login to wandb if not already done:"
    wandb login
fi

echo "🏃 Starting training in background..."
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "🎮 Detected $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "🚀 Using multi-GPU training with $GPU_COUNT GPUs"
    nohup python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT train.py > $LOG_FILE 2>&1 &
else
    echo "🚀 Using single GPU training"
    nohup python train.py > $LOG_FILE 2>&1 &
fi

TRAIN_PID=$!

echo "✨ Training started with PID: $TRAIN_PID"
echo "📊 Monitor at: https://wandb.ai"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "🔍 Tailing logs (Ctrl+C to stop tailing, training continues)..."
tail -f $LOG_FILE