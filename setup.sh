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
pip install wandb tqdm huggingface_hub

echo "📁 Creating directories..."
mkdir -p data models logs

echo "🔐 Setting up Hugging Face authentication..."
if [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN environment variable"
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "🔑 Hugging Face token required for Mistral-7B-Instruct access"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Also request access to: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
    echo ""
    read -p "Enter your Hugging Face token: " HF_TOKEN
    huggingface-cli login --token "$HF_TOKEN"
fi

echo "📥 Checking training data..."
if [ -f "./data/$DATA_FILE" ]; then
    echo "✅ Training data already exists: ./data/$DATA_FILE"
    echo "File size: $(du -h ./data/$DATA_FILE | cut -f1)"
else
    echo "📥 Downloading training data from $DATA_SOURCE..."
    scp $DATA_SOURCE ./data/$DATA_FILE
    echo "✅ Training data downloaded: ./data/$DATA_FILE"
fi

echo "🤖 Checking/downloading base model..."
if [ -d "./models/mistral-7b-instruct" ] && [ -f "./models/mistral-7b-instruct/config.json" ]; then
    echo "✅ Model already exists: ./models/mistral-7b-instruct"
else
    echo "📥 Downloading Mistral-7B-Instruct model..."
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading Mistral-7B-Instruct model...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_NAME',
    torch_dtype=torch.float16,
    device_map='cpu'
)

print('Saving model locally...')
tokenizer.save_pretrained('./models/mistral-7b-instruct')
model.save_pretrained('./models/mistral-7b-instruct')
print('✅ Model downloaded and saved!')
del model
torch.cuda.empty_cache()
"
fi

echo ""
echo "🔐 Setting up wandb..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "Using WANDB_API_KEY environment variable"
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "📊 WandB API key required for training monitoring"
    echo "Get your key from: https://wandb.ai/authorize"
    echo ""
    read -p "Enter your WandB API key: " WANDB_API_KEY
    wandb login --relogin "$WANDB_API_KEY"
fi
echo "✅ WandB authentication complete"

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