import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint

# Unsloth imports for faster training
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Wandb for experiment tracking
import wandb


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/sft"
    
    # Data settings
    train_data_path: str = "data/train_mul_direct.jsonl"
    val_data_path: Optional[str] = None
    max_seq_length: int = 512
    
    # Training settings
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 512
    eval_steps: Optional[int] = None
    save_total_limit: int = 3
    report_to: str = "wandb"  
    
    # Wandb settings
    wandb_project: Optional[str] = "banana-cot"
    wandb_run_name: Optional[str] = None
    
    # Unsloth/LoRA settings (optional, for even faster training)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    def __post_init__(self):
        """Set eval_steps based on save_steps if not provided."""
        if self.eval_steps is None:
            self.eval_steps = self.save_steps


def load_jsonl_dataset(path: str) -> list:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """Prepare dataset for training by formatting messages as text.
    
    SFTTrainer handles tokenization internally, so we just need to format
    the messages into text using the chat template.
    """
    data = load_jsonl_dataset(data_path)
    
    def format_function(examples):
        # Extract messages and apply chat template to create text
        messages = examples["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    # Format messages into text (SFTTrainer will tokenize)
    formatted_dataset = dataset.map(
        format_function,
        batched=False,
        remove_columns=[col for col in dataset.column_names if col != "messages"],
    )
    
    return formatted_dataset


def train_sft(config: SFTConfig):
    """Main training function for SFT using Unsloth for faster training."""
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if "wandb" in config.report_to.split(","):
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"sft-{Path(config.output_dir).name}",
            config={
                "model_name": config.model_name,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "max_seq_length": config.max_seq_length,
                "use_lora": config.use_lora,
            },
        )
    
    # Load model and tokenizer with Unsloth (much faster!)
    print(f"Loading model and tokenizer from {config.model_name} using Unsloth")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect: float16 for GPU, bfloat16 for Ampere+
        load_in_4bit=False,  # Set to True for QLoRA
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    # Optionally use LoRA for even faster training and lower memory
    if config.use_lora:
        print(f"Using LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=config.gradient_checkpointing,
            random_state=config.seed,
        )
    
    # Prepare datasets
    print(f"Loading training data from {config.train_data_path}")
    train_dataset = prepare_dataset(
        config.train_data_path,
        tokenizer,
        max_length=config.max_seq_length,
    )
    
    eval_dataset = None
    if config.val_data_path and os.path.exists(config.val_data_path):
        print(f"Loading validation data from {config.val_data_path}")
        eval_dataset = prepare_dataset(
            config.val_data_path,
            tokenizer,
            max_length=config.max_seq_length,
        )
    
    # Datasets are already formatted with "text" field from prepare_dataset
    train_dataset_text = train_dataset
    eval_dataset_text = eval_dataset
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="steps" if eval_dataset_text is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset_text is not None else False,
        metric_for_best_model="loss" if eval_dataset_text is not None else None,
        greater_is_better=False if eval_dataset_text is not None else None,
        report_to=config.report_to.split(",") if config.report_to else [],
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
    )
    
    # Initialize SFTTrainer (from TRL, works great with Unsloth)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_text,
        eval_dataset=eval_dataset_text,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )
    
    # Check for checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and training_args.do_train:
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    # Train
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    # Unsloth provides optimized saving methods
    if config.use_lora:
        # For LoRA models, save the LoRA adapters
        model.save_pretrained(config.output_dir)
        # Optionally save merged model for inference
        # model.save_pretrained_merged(config.output_dir, tokenizer, save_method="merged_16bit")
    else:
        # For full fine-tuning, save the entire model
        # Unsloth's save_pretrained is optimized
        model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    if eval_dataset_text is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    # Finish wandb run
    if "wandb" in config.report_to.split(","):
        wandb.finish()
    
    print(f"Training completed! Model saved to {config.output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model with SFT")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file (optional)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train_mul_direct.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation JSONL file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sft",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    
    args = parser.parse_args()
    
    # Load config from file or use defaults with CLI overrides
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = SFTConfig(**config_dict)
    else:
        config = SFTConfig()
    
    # Override with CLI arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
        config.per_device_eval_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length
    
    train_sft(config)


