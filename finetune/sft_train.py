import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Import unsloth first for optimizations
import unsloth
from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
from tqdm.auto import tqdm

from trl import SFTTrainer

import wandb

# Import evaluator functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evals.evaluator import extract_answer


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/sft"
    
    # Data settings
    train_data_path: str = "data/train_mul_direct.jsonl"
    val_data_path: Optional[str] = None  # For validation loss - should have messages format
    max_seq_length: int = 512
    
    # Math evaluation settings (separate from validation loss)
    math_eval_jsonl_path: Optional[str] = "evals/mul_eval_small.jsonl"  # For accuracy evaluation
    math_eval_max_new_tokens: int = 256
    math_eval_run_every_epoch: bool = True
    
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
    fp16: bool = False
    bf16: bool = True
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
    
    Handles two formats:
    1. Training format: {"messages": [...]}
    2. Eval format: {"prompt": "...", "answer": ...} - converts to messages format
    """
    data = load_jsonl_dataset(data_path)
    
    # Convert eval format to messages format if needed
    converted_data = []
    for ex in data:
        if "messages" in ex:
            # Already in training format
            converted_data.append(ex)
        elif "prompt" in ex:
            # Convert eval format to messages format (user only, no assistant response)
            # For validation, we'll just use the prompt without answer
            converted_data.append({
                "messages": [{"role": "user", "content": ex["prompt"]}]
            })
        else:
            raise ValueError(f"Unknown data format in {data_path}")
    
    def format_function(examples):
        messages = examples["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    
    dataset = Dataset.from_list(converted_data)
    
    formatted_dataset = dataset.map(
        format_function,
        batched=False,
        remove_columns=[col for col in dataset.column_names if col != "messages"],
    )
    
    return formatted_dataset


def eval_model_with_current_model(model, tokenizer, jsonl_path: str, max_new_tokens: int = 256, verbose: bool = False):
    """Evaluate model on dataset using the provided model and tokenizer (for training callbacks)."""
    correct = 0
    total = 0
    
    model.eval()
    device = next(model.parameters()).device
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
    
    print(f"Evaluating on {len(lines)} examples from {jsonl_path}")
    
    for idx, line in enumerate(tqdm(lines, desc="Math evaluation", leave=False)):
        try:
            ex = json.loads(line)
            prompt = ex["prompt"]
            gold = ex["answer"]

            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            
            gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            pred = extract_answer(gen)
            is_correct = pred is not None and pred == gold
            if is_correct:
                correct += 1
            total += 1
            
            # Log individual examples if verbose
            if verbose and (idx < 3 or not is_correct):  # Show first 3 or any incorrect
                status = "✓" if is_correct else "✗"
                print(f"  {status} Example {idx+1}: pred={pred}, gold={gold}, correct={is_correct}")
                if not is_correct and idx < 10:  # Show prediction for first few wrong ones
                    print(f"    Generated: {gen[:200]}...")
        except Exception as e:
            print(f"Error evaluating example {idx+1}: {e}")
            total += 1
    
    model.train()  # Set back to training mode
    return {"right": correct, "total": total}


class MathEvalCallback(TrainerCallback):
    """Callback to run math evaluation after each epoch and at training start."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.eval_jsonl_path = config.math_eval_jsonl_path
        self.max_new_tokens = config.math_eval_max_new_tokens
        self._initial_eval_done = False  # Track if initial eval has been done
        self._cached_model = None  # Cache model reference
        self._cached_tokenizer = None  # Cache tokenizer reference
        
    def _run_math_eval(self, model, tokenizer, epoch=None):
        """Helper method to run math evaluation."""
        # Check if we should run evaluation
        if not self.eval_jsonl_path:
            return
            
        # Check if eval dataset exists
        if not os.path.exists(self.eval_jsonl_path):
            print(f"Warning: Math eval dataset not found at {self.eval_jsonl_path}, skipping...")
            return
        
        # Get model and tokenizer if not provided
        if model is None or tokenizer is None:
            if hasattr(self, 'trainer'):
                # Get model (unwrap if needed for DDP/DataParallel)
                if model is None:
                    if hasattr(self.trainer, 'model'):
                        trainer_model = self.trainer.model
                        # Unwrap if wrapped (e.g., by DataParallel, DDP, etc.)
                        if hasattr(trainer_model, 'module'):
                            model = trainer_model.module
                        elif hasattr(self.trainer, 'accelerator') and hasattr(self.trainer.accelerator, 'unwrap_model'):
                            model = self.trainer.accelerator.unwrap_model(trainer_model)
                        else:
                            model = trainer_model
                
                # Get tokenizer
                if tokenizer is None:
                    if hasattr(self.trainer, 'tokenizer'):
                        tokenizer = self.trainer.tokenizer
                    elif hasattr(self.trainer, 'processing_class'):
                        tokenizer = self.trainer.processing_class
        
        if model is None or tokenizer is None:
            print("Warning: Model or tokenizer not available for math evaluation, skipping...")
            return
            
        epoch_label = f"epoch {int(epoch)}" if epoch is not None else "start"
        model_name = getattr(self.config, 'model_name', 'unknown')
        
        # Log what's being evaluated
        eval_source = "unknown"
        if model is not None:
            if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
                eval_source = f"checkpoint: {model.config.name_or_path}"
            elif hasattr(model, '__class__'):
                eval_source = f"model: {model.__class__.__name__}"
            else:
                eval_source = "training checkpoint"
        
        print(f"\n{'='*60}")
        print(f"Running math evaluation at {epoch_label}")
        print(f"Evaluating: {eval_source}")
        print(f"Base model: {model_name}")
        print(f"Dataset: {self.eval_jsonl_path}")
        print(f"{'='*60}")
        
        try:
            # Run evaluation with current training model
            eval_result = eval_model_with_current_model(
                model=model,
                tokenizer=tokenizer,
                jsonl_path=self.eval_jsonl_path,
                max_new_tokens=self.max_new_tokens,
                verbose=False  # Set to True for detailed per-example logging
            )
            
            # Extract accuracy
            accuracy = eval_result.get("right", 0) / eval_result.get("total", 1) if eval_result.get("total", 0) > 0 else 0.0
            
            # Log to wandb if enabled
            if "wandb" in self.config.report_to.split(","):
                log_dict = {
                    "math_eval/accuracy": accuracy,
                    "math_eval/correct": eval_result.get("right", 0),
                    "math_eval/total": eval_result.get("total", 0),
                }
                if epoch is not None:
                    log_dict["epoch"] = int(epoch)
                wandb.log(log_dict)
            
            print(f"Math Eval Results - {epoch_label.capitalize()}: {eval_result.get('right', 0)}/{eval_result.get('total', 0)} = {accuracy:.3f}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Error during math evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def on_train_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Run math evaluation at the start of training."""
        # Try to get model and tokenizer from trainer
        if model is None and hasattr(self, 'trainer') and hasattr(self.trainer, 'model'):
            model = self.trainer.model
        if tokenizer is None and hasattr(self, 'trainer') and hasattr(self.trainer, 'tokenizer'):
            tokenizer = self.trainer.tokenizer
            
        # Also check kwargs
        if model is None:
            model = kwargs.get('model')
        if tokenizer is None:
            tokenizer = kwargs.get('tokenizer')
        
        # Only run if we have model and tokenizer and haven't run initial eval yet
        if model is not None and tokenizer is not None and not self._initial_eval_done:
            self._run_math_eval(model, tokenizer, epoch=None)
            self._initial_eval_done = True
        
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Run math evaluation at the end of each epoch."""
        # Only run if enabled
        if not self.config.math_eval_run_every_epoch:
            return
        
        # PRIORITY 1: Get model and tokenizer from trainer - this is the CURRENT training checkpoint
        # The trainer should be set by the Trainer when callbacks are added
        # This is what we want - the model with updated weights from training
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Try to get unwrapped model (handles DDP/DataParallel wrapping)
            if model is None:
                if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                    trainer_model = self.trainer.model
                    # Unwrap if wrapped (e.g., by DataParallel, DDP, etc.)
                    if hasattr(trainer_model, 'module'):
                        model = trainer_model.module
                    elif hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
                        try:
                            model = self.trainer.accelerator.unwrap_model(trainer_model)
                        except:
                            model = trainer_model
                    else:
                        model = trainer_model
            
            # Get tokenizer from trainer
            if tokenizer is None:
                if hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer is not None:
                    tokenizer = self.trainer.tokenizer
                elif hasattr(self.trainer, 'processing_class') and self.trainer.processing_class is not None:
                    tokenizer = self.trainer.processing_class
        
        # PRIORITY 2: Try kwargs (if passed by Trainer)
        if model is None:
            model = kwargs.get('model')
        if tokenizer is None:
            tokenizer = kwargs.get('tokenizer')
        
        # Debug: Print what we have before calling _run_math_eval
        if model is None or tokenizer is None:
            debug_info = []
            debug_info.append(f"model from args: {model is not None}")
            debug_info.append(f"tokenizer from args: {tokenizer is not None}")
            debug_info.append(f"has trainer attr: {hasattr(self, 'trainer')}")
            
            if hasattr(self, 'trainer'):
                debug_info.append(f"trainer is not None: {self.trainer is not None}")
                if self.trainer is not None:
                    debug_info.append(f"has trainer.model: {hasattr(self.trainer, 'model')}")
                    debug_info.append(f"has trainer.tokenizer: {hasattr(self.trainer, 'tokenizer')}")
                    debug_info.append(f"has trainer.processing_class: {hasattr(self.trainer, 'processing_class')}")
                    if hasattr(self.trainer, 'model'):
                        debug_info.append(f"trainer.model is not None: {self.trainer.model is not None}")
                        if self.trainer.model is not None:
                            debug_info.append(f"trainer.model type: {type(self.trainer.model)}")
                    if hasattr(self.trainer, 'tokenizer'):
                        debug_info.append(f"trainer.tokenizer is not None: {self.trainer.tokenizer is not None}")
                    if hasattr(self.trainer, 'processing_class'):
                        debug_info.append(f"trainer.processing_class is not None: {self.trainer.processing_class is not None}")
                else:
                    debug_info.append("trainer is None!")
            else:
                debug_info.append("No trainer attribute!")
            
            print(f"DEBUG on_epoch_end: {'; '.join(debug_info)}")
            print(f"DEBUG kwargs keys: {list(kwargs.keys())}")
        
        # We MUST use the trainer's model (current checkpoint) - do not use cached model
        # The cached model is from BEFORE training and would give incorrect results
        
        # Run evaluation - _run_math_eval will also try to get them from trainer
        self._run_math_eval(model, tokenizer, epoch=state.epoch)
            

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
    
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
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
    
    train_dataset_text = train_dataset
    eval_dataset_text = eval_dataset
    
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
        eval_strategy="epoch" if eval_dataset_text is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset_text is not None else False,
        metric_for_best_model="loss" if eval_dataset_text is not None else None,
        greater_is_better=False if eval_dataset_text is not None else None,
        report_to=config.report_to.split(",") if config.report_to else [],
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
    )
    
    # Prepare callbacks
    callbacks = []
    if config.math_eval_run_every_epoch and config.math_eval_jsonl_path:
        math_eval_callback = MathEvalCallback(config=config)
        callbacks.append(math_eval_callback)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_text,
        eval_dataset=eval_dataset_text,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
        callbacks=callbacks,
    )
    
    # Run initial math evaluation before training starts (if callback is enabled)
    if config.math_eval_run_every_epoch and config.math_eval_jsonl_path:
        for callback in callbacks:
            if isinstance(callback, MathEvalCallback):
                print("\nRunning initial math evaluation before training...")
                # Store trainer reference - this is crucial for accessing model/tokenizer later
                callback.trainer = trainer
                # Store references to model and tokenizer in callback for later use
                callback._cached_model = model
                callback._cached_tokenizer = tokenizer
                callback._run_math_eval(model, tokenizer, epoch=None)
                callback._initial_eval_done = True
                break
    
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


