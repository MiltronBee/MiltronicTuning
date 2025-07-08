#!/usr/bin/env python3
import os
import json
import torch
import wandb
from datetime import datetime

# Disable DeepSpeed to avoid CUDA_HOME issues
os.environ["DISABLE_DEEPSPEED"] = "1"
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl_data(file_path):
    """Load JSONL training data with instruct format"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def format_instruction(item):
    """Format instruction data for Mistral"""
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    output = item.get('output', '')
    
    if input_text:
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {instruction} [/INST]"
    
    return f"{prompt} {output}"

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    try:
        # Check available GPUs
        device_count = torch.cuda.device_count()
        logger.info(f"🎮 Available CUDA devices: {device_count}")
        
        if device_count == 0:
            logger.error("No CUDA devices found! Training requires GPU.")
            return
    
        # Initialize wandb
        wandb.init(
            project="mistral-7b-finetune",
            name=f"mistral-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "dataset": "custom_instruct",
                "method": "LoRA",
                "gpu_count": device_count
            }
        )
    
        # Configuration
        model_path = "./models/mistral-7b-instruct"
        data_path = "./data/training_data.jsonl"
        output_dir = "./models/mistral-7b-finetuned"
    
        logger.info("🚀 Starting Mistral-7B fine-tuning...")
        
        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
        # Load and prepare data
        logger.info("Loading training data...")
        raw_data = load_jsonl_data(data_path)
        
        # Format data
        formatted_data = []
        for item in raw_data:
            formatted_text = format_instruction(item)
            formatted_data.append({"text": formatted_text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        train_size = int(0.9 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            max_steps=1000,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            run_name=f"mistral-finetune-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataloader_num_workers=4,
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_pin_memory=True,
            group_by_length=True,
        )
    
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    
        # Start training
        logger.info("🏃 Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapter
        model.save_pretrained(f"{output_dir}/lora_adapter")
        
        logger.info("✅ Training completed successfully!")
        wandb.finish()
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        raise

if __name__ == "__main__":
    main()