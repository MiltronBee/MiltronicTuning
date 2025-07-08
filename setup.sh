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
# Load existing .env if it exists
if [ -f ".env" ]; then
    source .env
fi

if [ -n "$HF_TOKEN" ]; then
    echo "Using existing HF_TOKEN"
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "🔑 Hugging Face token required for Mistral-7B-Instruct access"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Also request access to: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
    echo ""
    read -p "Enter your Hugging Face token: " HF_TOKEN
    
    # Save to .env file
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo "✅ Token saved to .env file"
    
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
if [ -d "./models/mistral-7b-instruct" ] && [ -f "./models/mistral-7b-instruct/config.json" ] && [ -f "./models/mistral-7b-instruct/pytorch_model.bin" -o -f "./models/mistral-7b-instruct/pytorch_model-00001-of-00002.bin" ]; then
    echo "✅ Model already exists: ./models/mistral-7b-instruct"
    echo "Model size: $(du -sh ./models/mistral-7b-instruct | cut -f1)"
else
    echo "📥 Downloading Mistral-7B-Instruct model..."
    python -c "
from huggingface_hub import snapshot_download
import os

print('Downloading Mistral-7B-Instruct model...')
snapshot_download(
    repo_id='$MODEL_NAME',
    local_dir='./models/mistral-7b-instruct',
    local_dir_use_symlinks=False
)
print('✅ Model downloaded and saved!')
"
fi

echo ""
echo "🔐 Setting up wandb..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "Using existing WANDB_API_KEY"
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "📊 WandB API key required for training monitoring"
    echo "Get your key from: https://wandb.ai/authorize"
    echo ""
    read -p "Enter your WandB API key: " WANDB_API_KEY
    
    # Append to .env file
    echo "WANDB_API_KEY=$WANDB_API_KEY" >> .env
    echo "✅ WandB key saved to .env file"
    
    wandb login --relogin "$WANDB_API_KEY"
fi
echo "✅ WandB authentication complete"

echo "🏃 Starting training in background..."
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "🎮 Detected $GPU_COUNT GPU(s)"

# Set environment variables to disable DeepSpeed
export DISABLE_DEEPSPEED=1

# For debugging, let's start with single GPU and check logs
echo "🚀 Starting training (single GPU for debugging)"
echo "📝 Check logs with: tail -f $LOG_FILE"
nohup python train.py > $LOG_FILE 2>&1 &

TRAIN_PID=$!

echo "✨ Training started with PID: $TRAIN_PID"
echo "📊 Monitor at: https://wandb.ai"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "🔍 Tailing logs (Ctrl+C to stop tailing, training continues)..."
tail -f $LOG_FILE